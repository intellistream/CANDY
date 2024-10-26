/*
 * Copyright (C) 2024 by the INTELLI team
 * Created on: 2024/10/26 10:48
 * Description: ${DESCRIPTION}
 */

#ifndef ANNS_ALGORITHM_BASE_HPP
#define ANNS_ALGORITHM_BASE_HPP

#include <torch/torch.h>
#include <vector>
#include <string>
#include <Algorithms/AbstractANNSAlgorithm.hpp>
#include <Utils/ConfigMap.hpp>

class ANNSBase : public CANDY::AbstractANNS {
public:
    virtual ~ANNSBase() = default;

    void reset() override;

    bool startHPC() override;

    bool endHPC() override;

    bool setConfig(INTELLI::ConfigMapPtr cfg) override = 0;

    bool insertTensor(torch::Tensor &t) override = 0;

    bool loadInitialTensor(torch::Tensor &t) override = 0;

    bool deleteTensor(torch::Tensor &t, int64_t k) override = 0;

    bool reviseTensor(torch::Tensor &t, torch::Tensor &w) override = 0;

    bool resetIndexStatistics() override;

    INTELLI::ConfigMapPtr getIndexStatistics() override;
};

#endif // ANNS_ALGORITHM_BASE_HPP
