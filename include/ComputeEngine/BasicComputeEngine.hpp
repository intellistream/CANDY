/*
* Copyright (C) 2024 by the jjzhao
 * Created on: 2024/11/24
 * Description: [Provide description here]
 */
#ifndef CANDY_INCLUDE_ComputeEngine_BasicComputeEngine_H_
#define CANDY_INCLUDE_ComputeEngine_BasicComputeEngine_H_
#include <torch/torch.h>
#include <vector>
#include <cmath>  // For std::sqrt
#include <numeric>
#include <ComputeEngine/AbstractComputeEngine.hpp>
class BasicComputeEngine: public AbstractComputeEngine {
  public:
   BasicComputeEngine();
   ~BasicComputeEngine() override;
   float euclidean_distance(const torch::Tensor& a, const torch::Tensor& b) override;
   float computeL2Distance(const float* a, const float* b, const size_t size);
   float computeL2Distance(const std::vector<float>& a, const std::vector<float>& b);
   float euclidean_distance(const std::vector<float>& a, const std::vector<float>& b);
};
#endif  //CANDY_INCLUDE_ComputeEngine_BasicComputeEngine_H_