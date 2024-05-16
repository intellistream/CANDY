//
// Created by tony on 16/05/24.
//
#include <CANDY/YingYangHNSWIndex.h>
bool CANDY::YinYangHNSWIndex::setConfig(INTELLI::ConfigMapPtr cfg) {
  AbstractIndex::setConfig(cfg);
  if (faissMetric != faiss::METRIC_INNER_PRODUCT) {
    INTELLI_WARNING("I can only deal with inner product distance");
  }
  vecDim = cfg->tryI64("vecDim", 768, true);
  maxHNSWVolume = cfg->tryI64("maxHNSWVolume", 1000000, true);
  maxConnection = cfg->tryI64("maxConnection", 32, true);
  efConstruction = cfg->tryI64("efConstruction", 200, true);
  inlineCfg = newConfigMap();
  inlineCfg->loadFrom(cfg.get()[0]);
  inlineCfg->edit("initialVolume",(int64_t)1);
  inlineCfg->edit("DCOBatchSize",(int64_t )128);
  hnswlib::InnerProductSpace space(vecDim);
  alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, maxHNSWVolume, maxConnection, efConstruction);
  return true;
}
bool CANDY::YinYangHNSWIndex::loadInitialTensor(torch::Tensor &t) {
  /**
   * @brief just to make each row as vertex in HNSW
   */
  auto n = t.size(0);
  for (int64_t i = 0; i < n; i++) {
    auto tCopy = t.slice(0, i, i + 1).clone().contiguous();
    YinYangHNSW_YinVertex *ver=new YinYangHNSW_YinVertex;
    ver->verTensor=tCopy;
    ver->yangIndex.setConfig(inlineCfg);
    ver->yangIndex.insertTensor(tCopy);
    auto idx = reinterpret_cast<long>(ver);
    alg_hnsw->addPoint(tCopy.data_ptr<float>(), idx);
  }
  return true;
}
bool CANDY::YinYangHNSWIndex::insertTensor(torch::Tensor &t) {
  auto n = t.size(0);
  for (int64_t i = 0; i < n; i++) {
    auto tCopy = t.slice(0, i, i + 1).clone().contiguous();
    std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(tCopy.data_ptr<float>(), 1);
    auto ver =  reinterpret_cast<YinYangHNSW_YinVertex *>(*reinterpret_cast<long *>(result.top().second));
    ver->yangIndex.insertTensor(tCopy);
  }
  return true;
}
torch::Tensor CANDY::YinYangHNSWIndex::searchRow(torch::Tensor &q, int64_t k) {
  torch::Tensor ru=torch::zeros({k, vecDim});
  std::priority_queue<std::pair<float, hnswlib::labeltype>> hnswRu = alg_hnsw->searchKnn(q.contiguous().data_ptr<float>(), k);
  int64_t copiedResults=0;
  while ((!hnswRu.empty())&&(copiedResults<k)) {
    auto ver =  reinterpret_cast<YinYangHNSW_YinVertex *>(*reinterpret_cast<long *>(hnswRu.top().second));
    int64_t copyThisTime = ver->yangIndex.size();
    if(copyThisTime+copiedResults>k) {
      copyThisTime=k- copiedResults;
    }
    ru.slice(0,copiedResults,copiedResults+copyThisTime)=ver->yangIndex.searchTensor(q,copyThisTime)[0];
    copiedResults+=copyThisTime;
    hnswRu.pop();
  }
 return ru;
}
std::vector<torch::Tensor> CANDY::YinYangHNSWIndex::searchTensor(torch::Tensor &q, int64_t k) {
  size_t tensors = (size_t) q.size(0);
  std::vector<torch::Tensor> ru(tensors);
  for (int64_t i = 0; i < tensors; i++) {
    auto tCopy = q.slice(0, i, i + 1);
    ru[i] =searchRow(tCopy,k);
  }
  return ru;

}