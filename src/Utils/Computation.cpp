/*
* Copyright (C) 2024 by the INTELLI team
 * Created on: 2024/10/14
 * Description: [Provide description here]
 */
#include <vector>
#include <Utils/Computation.hpp>
#include <cmath>  // Added for std::sqrt
#include <numeric>
#include <functional>
// Utility function to calculate Euclidean distance between two vectors
float Computation::euclidean_distance(const std::vector<float> &a, const std::vector<float> &b) {
 float sum = 0.0;
 for (size_t i = 0; i < a.size(); ++i) {
  float diff = a[i] - b[i];
  sum += diff * diff;
 }
 return std::sqrt(sum);
}

float Computation::computeL2Distance(const float *a, const float *b, size_t size) {
 return std::inner_product(a, a + size, b, 0.0f, std::plus<float>(),
                           [](float x, float y) { return (x - y) * (x - y); });
}

