/*
* Copyright (C) 2024 by the jjzhao
 * Created on: 2024/11/24
 * Description: [Provide description here]
 */
#ifndef SEPEARATE_ANNS_ALGORITHM_BASE_HPP
#define SEPEARATE_ANNS_ALGORITHM_BASE_HPP
#include <torch/torch.h>
#include <vector>
#include <Algorithms/AbstractSeparateANNSAlgorithm.hpp>
#include <IO/BasicStorage.hpp>

class SeparateANNSBase: public AbstractSeparateANNSAlgorithm{
 public:
  BasicStorage storage_engine;
  SeparateANNSBase();
  ~SeparateANNSBase() override;
  bool insertTensor(const torch::Tensor &t) override;
  bool insertTensorWithRawId(const torch::Tensor &t, int rowId) override;
  std::vector<int> searchTensor(const torch::Tensor &t, int64_t k) override;
  std::vector<int> deleteTensor(const torch::Tensor &t, int64_t k) override;
  std::vector<int> findKnnTensor(const torch::Tensor &t, int64_t k) override;
};
#endif  // SEPEARATE_ANNS_ALGORITHM_BASE_HPP