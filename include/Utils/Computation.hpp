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

namespace CANDY {
inline float computeL2Distance(const float* a, const float* b,
                               const size_t size) {
  return std::inner_product(
      a, a + size, b, 0.0f, std::plus<float>(),
      [](const float x, const float y) { return (x - y) * (x - y); });
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
  return torch::norm(a - b).item<float>();
}
}  // namespace CANDY
#endif  // COMPUTATION_H
