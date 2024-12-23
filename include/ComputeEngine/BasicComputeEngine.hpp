/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Jianjun Zhao
 * Created on: 2024/12/21
 * Description: Compute Engine for Basic Compute Engine
 */
#ifndef CANDY_INCLUDE_ComputeEngine_BasicComputeEngine_H_
#define CANDY_INCLUDE_ComputeEngine_BasicComputeEngine_H_
#include <torch/torch.h>
#include <ComputeEngine/AbstractComputeEngine.hpp>
#include <Utils/Computation.hpp>

namespace CANDY_COMPUTE {
class BasicComputeEngine : public AbstractComputeEngine {
 public:
  BasicComputeEngine();
  ~BasicComputeEngine() override;
  float euclidean_distance(const torch::Tensor& a,
                           const torch::Tensor& b) override;
  torch::Tensor pairwise_euclidean_distance(torch::Tensor A, torch::Tensor B,
                                            CANDY::AMMTYPE ammtype,
                                            int64_t sketchsize) override;
  torch::Tensor pairwise_euclidean_distance(torch::Tensor A, torch::Tensor B,
                                            CANDY::AMMTYPE ammtype,
                                            int64_t sketchsize,
                                            torch::Tensor B_norm) override;
};
}  // namespace CANDY_COMPUTE
#endif  //CANDY_INCLUDE_ComputeEngine_BasicComputeEngine_H_