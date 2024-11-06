/*
 * Copyright (C) 2024 by the INTELLI team
 * Created on: 24-11-5 下午2:07
 * Description: ${DESCRIPTION}
 */

#ifndef LSHSEARCH_HPP
#define LSHSEARCH_HPP

#include <torch/torch.h>
#include <Algorithms/ANNSBase.hpp>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

namespace CANDY_ALGO {
class LshSearch : public ANNSBase {
 protected:
  INTELLI::ConfigMapPtr myCfg = nullptr;

 public:
  ~LshSearch() override = default;

  // Constructor with vector dimensions
  explicit LshSearch(size_t Dimensions, size_t NumPlanes);

  bool setConfig(INTELLI::ConfigMapPtr cfg) override;

  void reset() override;

  bool insertTensor(const torch::Tensor& t) override;

  bool deleteTensor(torch::Tensor& t, int64_t k = 1) override;

  bool reviseTensor(torch::Tensor& t, torch::Tensor& w) override;

  std::vector<torch::Tensor> searchTensor(const torch::Tensor& q,
                                          int64_t k) override;

 private:
  size_t Dimensions;
  size_t GlobalIndexCounter = 0;

  // Hash table (unordered_map) where each bucket corresponds to a map of tensors
  std::unordered_map<size_t, std::unordered_map<size_t, torch::Tensor>> Index;

  // Store hyperplane information
  std::vector<torch::Tensor> RandomHyperplanes;

  // Generate hyperplanes
  void GenerateRandomHyperplanes(size_t NumPlanes);

  size_t HashFunction(const torch::Tensor& t);
};
}  // namespace CANDY_ALGO

typedef std::shared_ptr<CANDY_ALGO::LshSearch> LSHSearchPtr;

#endif  //LSHSEARCH_HPP
