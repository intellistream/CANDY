/*
* Copyright (C) 2024 by the INTELLI team
 * Created by: Ziao Wang
 * Created on: 2024/10/22
 * Description: [Provide description here]
 */

#ifndef CANDY_INCLUDE_ALGORITHMS_UTILS_METRIC_TYPE_HPP
#define CANDY_INCLUDE_ALGORITHMS_UTILS_METRIC_TYPE_HPP
#include <cstdint>
/// Most algorithms support both inner product and L2, with the flat
/// (brute-force) indices supporting additional metric types for vector
/// comparison.
enum MetricType {
  METRIC_INNER_PRODUCT = 0,
  METRIC_L2 = 1,
};

/// all vector indices are this type
using idx_t = int64_t;

#endif //CANDY_INCLUDE_ALGORITHMS_UTILS_METRIC_TYPE_HPP
