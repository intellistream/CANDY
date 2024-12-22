/*
* Copyright (C) 2024 by the jjzhao
 * Created on: 2024/11/24
 * Description: [Provide description here]
 */

#include <Algorithms/SeparateANNSBase.hpp>
namespace CANDY_ALGO {
SeparateANNSBase::SeparateANNSBase() {
  this -> storage_engine = std::make_shared<BasicStorage>();
}
SeparateANNSBase::~SeparateANNSBase() {

}
bool SeparateANNSBase::insertTensor(const torch::Tensor &t) {
  return storage_engine.insertTensor(t);
}
bool SeparateANNSBase::insertTensorWithRawId(const torch::Tensor &t, int rowId) {
  return storage_engine.insertTensorWithRawId(t, rowId);
}
std::vector<int> SeparateANNSBase::searchTensor(const torch::Tensor &t, int64_t k) {
  std::vector<int> rowIds(k, 0);
  std::vector<int> k_vectorId = findKnnTensor(t, k);
  for (int i = 0; i < k; i++) {
    rowIds[i] = storage_engine.getRawIdByVid(k_vectorId[i]);
  }
  INTELLI_INFO("Search the tensors in vector storage successfully: " + std::to_string(k));
  return rowIds;
}
std::vector<int> SeparateANNSBase::deleteTensor(const torch::Tensor &t, int64_t k) {
  std::vector<int> vidsToDelete = findKnnTensor(t, k);
  std::vector<int> rowIdsToDelete(k, 0);
  rowIdsToDelete = storage_engine.deleteTensor(vidsToDelete);
  INTELLI_INFO("Delete the tensors in vector storage successfully");
  return rowIdsToDelete;
}
std::vector<int> SeparateANNSBase::findKnnTensor(const torch::Tensor &t, int64_t k) {
  std::vector<int> k_vectorId(k, 0);
  //TODO: search the k-NN of a query tensor, return the result tensors
  for (int i = 0; i < k; i++) {
    k_vectorId[i] = i;
  }
  return k_vectorId;
}
}



