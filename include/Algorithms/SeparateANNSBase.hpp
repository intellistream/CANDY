/*
* Copyright (C) 2024 by the jjzhao
 * Created on: 2024/11/24
 * Description: [Provide description here]
 */
#ifndef SPEARATE_ANNS_ALGORITHM_BASE_HPP
#define SPEARATE_ANNS_ALGORITHM_BASE_HPP
#include <torch/torch.h>
#include <vector>
#include "AbstractSeparateANNSAlgorithm.hpp"

class SeparateANNSBase: public AbstractSeparateANNSAlgorithm{
 public:
  SeparateANNSBase();
  ~SeparateANNSBase();
  std::vector<int> searchTensor(const torch::Tensor &t, int64_t k);
  std::vector<int> deleteTensor(const torch::Tensor &t, int64_t k);
  std::vector<int> findKnnTensor(const torch::Tensor &t, int64_t k);
};
#endif  // SPEARATE_ANNS_ALGORITHM_BASE_HPP