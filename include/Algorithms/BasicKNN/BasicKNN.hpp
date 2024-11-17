/*
* Copyright (C) 2024 by the INTELLI team
 * Created on: 2024/10/9
 * Description: [Provide description here]
 */

#ifndef CANDY_INCLUDE_ALGORITHMS_BASIC_KNN_HPP_
#define CANDY_INCLUDE_ALGORITHMS_BASIC_KNN_HPP_

#include <torch/torch.h>
#include <Algorithms/ANNSBase.hpp>
#include <memory>
#include <unordered_map>
#include <vector>

namespace CANDY_ALGO {

class BasicKNN : public ANNSBase {
protected:
  INTELLI::ConfigMapPtr myCfg = nullptr;
  torch::Tensor dbTensor;
  int64_t lastNNZ = 0;
  int64_t vecDim = 0, initialVolume = 1000, expandStep = 100;

public:
  // Destructor
  ~BasicKNN() override = default;

  BasicKNN() {}

  // Constructor with vector dimensions
  BasicKNN(size_t dimensions);

  bool setConfig(INTELLI::ConfigMapPtr cfg) override;

  void reset() override;

  bool insertTensor(const torch::Tensor& t) override;

  bool deleteTensor(torch::Tensor& t, int64_t k = 1) override;

  bool reviseTensor(torch::Tensor& t, torch::Tensor& w) override;

  std::vector<torch::Tensor> searchTensor(const torch::Tensor& q,
                                          int64_t k) override;

private:
  size_t dimensions;
  std::unordered_map<size_t, torch::Tensor> index;
};

typedef std::shared_ptr<BasicKNN> BasicKNNPtr;
#define newKNNIndex std::make_shared<CANDY_ALGO::BasicKNN>
}  // namespace CANDY_ALGO

#endif
