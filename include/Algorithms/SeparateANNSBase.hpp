/*
* Copyright (C) 2024 by the INTELLI team
 * Created by: Jianjun Zhao
 * Created on: 2024/12/21
 * Description:
 */
#ifndef SEPEARATE_ANNS_ALGORITHM_BASE_HPP
#define SEPEARATE_ANNS_ALGORITHM_BASE_HPP
#include <torch/torch.h>
#include <Algorithms/AbstractSeparateANNSAlgorithm.hpp>
#include <Utils/TimeStampGenerator.hpp>
#include <vector>

namespace CANDY_ALGO {
class SeparateANNSBase: public AbstractSeparateANNSAlgorithm {
 public:
  SeparateANNSBase();
  ~SeparateANNSBase() override;
  bool insertTensor(const torch::Tensor &t) override;
  std::vector<torch::Tensor> searchTensor(const torch::Tensor &t, int64_t k) override;
  std::vector<torch::Tensor> deleteTensor(const torch::Tensor &t, int64_t k) override;
  bool reviseTensor(const torch::Tensor &t, const torch::Tensor &w) override;
  std::vector<int64_t> findKnnTensor(const torch::Tensor &t, int64_t k) override;
  bool setConfig(INTELLI::ConfigMapPtr cfg) override;
};
typedef std::shared_ptr<SeparateANNSBase> SeparateANNSBasePtr;
}  // namespace CANDY_ALGO
#endif  // SEPEARATE_ANNS_ALGORITHM_BASE_HPP