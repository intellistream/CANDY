/*
* Copyright (C) 2024 by the INTELLI team
 * Created on: 2024/10/14
 * Description: [Provide description here]
 */
#ifndef COMPUTATION_H
#define COMPUTATION_H


class Computation {
public:
    static float euclidean_distance(const std::vector<float> &a, const std::vector<float> &b);

    static float computeL2Distance(const float *a, const float *b, size_t size);
};


#endif //COMPUTATION_H
