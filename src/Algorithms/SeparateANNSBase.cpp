/*
* Copyright (C) 2024 by the jjzhao
 * Created on: 2024/11/24
 * Description: [Provide description here]
 */

#include <Algorithms/SeparateANNSBase.hpp>
#include <IO/BasicStorage.hpp>
namespace CANDY_ALGO {
SeparateANNSBase::SeparateANNSBase() {
  this -> storage_engine = std::make_shared<CANDY_STORAGE::BasicStorage>();
}
SeparateANNSBase::~SeparateANNSBase() {

}
bool SeparateANNSBase::insertTensor(const torch::Tensor &t) {
  return storage_engine -> insertTensor(t);
}
std::vector<torch::Tensor> SeparateANNSBase::searchTensor(const torch::Tensor &t, int64_t k) {
  return storage_engine -> getVectorByVids(findKnnTensor(t, k));
}
std::vector<torch::Tensor> SeparateANNSBase::deleteTensor(const torch::Tensor &t, int64_t k) {
  return storage_engine -> deleteTensor(findKnnTensor(t, k));
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



