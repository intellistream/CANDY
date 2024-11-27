/*
* Copyright (C) 2024 by the jjzhao
 * Created on: 2024/11/25
 * Description: [Provide description here]
 */
#include <ComputeEngine/BasicComputeEngine.hpp>

BasicComputeEngine::BasicComputeEngine() {

}
BasicComputeEngine::~BasicComputeEngine() {

}
float BasicComputeEngine::computeL2Distance(const float* a, const float* b,
                               const size_t size) {
  static auto func = [](float x, float y) {
    float diff = x - y;
    return diff * diff;
  };
  return std::inner_product(
      a, a + size, b, 0.0f, std::plus<float>(),
      func);  // std::plus<float>() is the binary function object that will be applied.
}

float BasicComputeEngine::computeL2Distance(const std::vector<float>& a,
                               const std::vector<float>& b) {
  return computeL2Distance(a.data(), b.data(), a.size());
}

float BasicComputeEngine::euclidean_distance(const std::vector<float>& a,
                                const std::vector<float>& b) {
  return std::sqrt(computeL2Distance(a, b));
}

float BasicComputeEngine::euclidean_distance(const torch::Tensor& a,
                                const torch::Tensor& b) {
  return computeL2Distance(a.data_ptr<float>(), b.data_ptr<float>(), a.size(0));
}