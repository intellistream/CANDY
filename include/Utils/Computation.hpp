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
    inline float computeL2Distance(const float *a, const float *b, size_t size) {
        return std::inner_product(a, a + size, b, 0.0f, std::plus<float>(),
                                  [](float x, float y) { return (x - y) * (x - y); });
    }
}
#endif // COMPUTATION_H
