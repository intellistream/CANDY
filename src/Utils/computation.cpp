/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/16
 * Modified by:
 * Modified on:
 * Description: [Provide description here]
 */

#include <Utils/computation.hpp>
#include <cmath>  // Added for std::sqrt
#include <vector>
// Utility function to calculate Euclidean distance between two vectors
float computation::euclidean_distance(const std::vector<float>& a,
                                      const std::vector<float>& b) {
  float sum = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return std::sqrt(sum);
}
