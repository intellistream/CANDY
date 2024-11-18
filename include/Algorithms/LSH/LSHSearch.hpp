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

class LSHSearch : public ANNSBase {
 protected:
  INTELLI::ConfigMapPtr myCfg = nullptr;

 public:
  LSHSearch() = default;

  ~LSHSearch() override = default;

  bool setConfig(INTELLI::ConfigMapPtr cfg) override;

  void reset() override;

  bool insertTensor(const torch::Tensor& t) override;

  bool deleteTensor(torch::Tensor& t, int64_t k = 1) override;

  bool reviseTensor(torch::Tensor& t, torch::Tensor& w) override;

  std::vector<torch::Tensor> searchTensor(const torch::Tensor& q,
                                          int64_t k) override;

 private:
  size_t Dimensions;
  size_t NumofHyperplanes;
  size_t GlobalIndexCounter = 0;

  // Hash table (unordered_map) where each bucket corresponds to a map of tensors
  std::unordered_map<std::string, std::unordered_map<int64_t, torch::Tensor>>
      Index;

  // Store hyperplane information
  std::vector<torch::Tensor> RandomHyperplanes;


  std::unordered_map<int64_t, std::string> idToBucket;

  // Generate random hyperplanes for hashing
  void GenerateRandomHyperplanes(size_t NumPlanes);

  // Hash function to map tensors to hash buckets
  std::string HashFunction(const torch::Tensor& t);

  // Hamming distance calculation for two binary strings
  int HammingDistance(const std::string& str1, const std::string& str2);
};

}  // namespace CANDY_ALGO

typedef std::shared_ptr<CANDY_ALGO::LSHSearch> LSHSearchPtr;

#endif  // LSHSEARCH_HPP
