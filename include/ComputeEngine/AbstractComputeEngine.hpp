/*
* Copyright (C) 2024 by the jjzhao
 * Created on: 2024/11/24
 * Description: [Provide description here]
 */
#ifndef CANDY_INCLUDE_ComputeEngine_AbstractComputeEngine_H_
#define CANDY_INCLUDE_ComputeEngine_AbstractComputeEngine_H_
#include <torch/torch.h>
class AbstractComputeEngine{
  public:
   AbstractComputeEngine() = default;
   virtual ~AbstractComputeEngine() = default;
   virtual float euclidean_distance(const torch::Tensor& a, const torch::Tensor& b) = 0;
};
#endif  //CANDY_INCLUDE_ComputeEngine_AbstractComputeEngine_H_