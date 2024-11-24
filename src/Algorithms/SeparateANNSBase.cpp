/*
* Copyright (C) 2024 by the jjzhao
 * Created on: 2024/11/24
 * Description: [Provide description here]
 */
// ReSharper disable All
#include <torch/torch.h>
#include <../include/Algorithms/SeparateANNSBase.hpp>
#include <vector>

#include "IO/BasicStorage.hpp"

SeparateANNSBase::SeparateANNSBase() {
  storage_engine = BasicStorage();
}
SeparateANNSBase::~SeparateANNSBase() {

}
std::vector<int> SeparateANNSBase::searchTensor(const torch::Tensor &t, int64_t k) {
  vector<int> rowIds(k, 0);
  vector<int> k_vectorId = findKnnTensor(t, k);
  for (int i = 0; i < k; i++) {
    rowIds[i] = storage_engine.getRowIdByVid(k_vectorId[i]);
  }
  return rowIds;
}
std::vector<int> SeparateANNSBase::deleteTensor(const torch::Tensor &t, int64_t k) {
  vector<int> vidsToDelete = findKnnTensor(t, k);
  vector<int> rowIdsToDelete(k, 0);
  rowIdsToDelete = storage_engine.deleteTensor(vidsToDelete);
  return rowIdsToDelete;
}
std::vector<int> SeparateANNSBase::findKnnTensor(const torch::Tensor &t, int64_t k) {
  vector<int> k_vectorId(k, 0);
  //TODO: search the k-NN of a query tensor, return the result tensors
  for (int i = 0; i < k; i++) {
    k_vectorId[i] = i;
  }
}



