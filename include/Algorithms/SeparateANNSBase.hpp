/*
* Copyright (C) 2024 by the INTELLI team
 * Created by: Jianjun Zhao
 * Created on: 2024/12/21
 * Description:
 */
#ifndef SEPEARATE_ANNS_ALGORITHM_BASE_HPP
#define SEPEARATE_ANNS_ALGORITHM_BASE_HPP
#include <torch/torch.h>
#include <vector>
#include <Algorithms/AbstractSeparateANNSAlgorithm.hpp>
#include <IO/BasicStorage.hpp>

namespace CANDY_ALGO {
class SeparateANNSBase;
typedef std::shared_ptr<SeparateANNSBase> SeparateANNSBasePtr;
class SeparateANNSBase: public CANDY::AbstractSeparateANNSAlgorithm{
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
}  // namespace CANDY_ALGO
#endif  // SEPEARATE_ANNS_ALGORITHM_BASE_HPP