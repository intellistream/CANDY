/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Jianjun Zhao
 * Created on: 2024/12/21
 * Description:
 */
#include <ComputeEngine/BasicComputeEngine.hpp>

namespace CANDY_COMPUTE {
BasicComputeEngine::BasicComputeEngine() {}

BasicComputeEngine::~BasicComputeEngine() {}

float BasicComputeEngine::euclidean_distance(const torch::Tensor& a,
                                             const torch::Tensor& b) {
  return CANDY::computeL2Distance(a.data_ptr<float>(), b.data_ptr<float>(),
                                  a.size(0));
}

torch::Tensor BasicComputeEngine::pairwise_euclidean_distance(
    torch::Tensor A, torch::Tensor B, CANDY::AMMTYPE ammtype,
    int64_t sketchsize) {
  return CANDY::pairwise_euclidean_distance(A, B, ammtype, sketchsize);
}

torch::Tensor BasicComputeEngine::pairwise_euclidean_distance(
    torch::Tensor A, torch::Tensor B, CANDY::AMMTYPE ammtype,
    int64_t sketchsize, torch::Tensor B_norm) {
  return CANDY::pairwise_euclidean_distance(A, B, ammtype, sketchsize, B_norm);
}
}  // namespace CANDY_COMPUTE