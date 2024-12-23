/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Jianjun Zhao
 * Created on: 2024/12/21
 * Description: ComputeEngine is a module that provides the basic computation functions.
 */
#ifndef CANDY_INCLUDE_ComputeEngine_AbstractComputeEngine_H_
#define CANDY_INCLUDE_ComputeEngine_AbstractComputeEngine_H_
#include "Utils/Computation.hpp"

#include <torch/torch.h>

namespace CANDY_COMPUTE {
class AbstractComputeEngine {
 public:
  AbstractComputeEngine() = default;
  virtual ~AbstractComputeEngine() = default;
  virtual float euclidean_distance(const torch::Tensor& a,
                                   const torch::Tensor& b) = 0;
  virtual torch::Tensor pairwise_euclidean_distance(torch::Tensor A,
                                                    torch::Tensor B,
                                                    CANDY::AMMTYPE ammtype,
                                                    int64_t sketchsize) = 0;
  virtual torch::Tensor pairwise_euclidean_distance(torch::Tensor A,
                                                    torch::Tensor B,
                                                    CANDY::AMMTYPE ammtype,
                                                    int64_t sketchsize,
                                                    torch::Tensor B_norm) = 0;
};

typedef std::shared_ptr<AbstractComputeEngine> AbstractComputeEnginePtr;
}  // namespace CANDY_COMPUTE
#endif  //CANDY_INCLUDE_ComputeEngine_AbstractComputeEngine_H_