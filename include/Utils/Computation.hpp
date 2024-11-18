/*
* Copyright (C) 2024 by the INTELLI team
 * Created on: 2024/10/14
 * Description: [Provide description here]
 */
#ifndef COMPUTATION_H
#define COMPUTATION_H

#include <cmath>  // For std::sqrt
#include <numeric>
#include <vector>

#include <torch/torch.h>

namespace CANDY {
inline float computeL2Distance(const float* a, const float* b,
                               const size_t size) {
  static auto func = [](float x, float y) {
    float diff = x - y;
    return diff * diff;
  };
  return std::inner_product(
      a, a + size, b, 0.0f, std::plus<float>(),
      func);  // std::plus<float>() is the binary function object that will be applied.
}

inline float computeL2Distance(const std::vector<float>& a,
                               const std::vector<float>& b) {
  return computeL2Distance(a.data(), b.data(), a.size());
}

inline float euclidean_distance(const std::vector<float>& a,
                                const std::vector<float>& b) {
  return std::sqrt(computeL2Distance(a, b));
}

inline float euclidean_distance(const torch::Tensor& a,
                                const torch::Tensor& b) {
  return computeL2Distance(a.data_ptr<float>(), b.data_ptr<float>(), a.size(0));
}
}  // namespace CANDY
#endif  // COMPUTATION_H
