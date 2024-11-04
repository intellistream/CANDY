/*
 * Copyright (C) 2024 by the INTELLI team
 * Created on: 2024/10/26 10:48
 * Description: ${DESCRIPTION}
 */

#ifndef ANNS_ALGORITHM_BASE_HPP
#define ANNS_ALGORITHM_BASE_HPP

#include <torch/torch.h>
#include <vector>
#include <Algorithms/AbstractANNSAlgorithm.hpp>
#include <Utils/ConfigMap.hpp>
#include <Utils/Param.hpp>

class ANNSBase;
typedef std::shared_ptr<ANNSBase> ANNSBasePtr;

class ANNSBase : public CANDY::AbstractANNS {
public:
    virtual ~ANNSBase() = default;

    // Methods with default implementations
    // Methods with default implementations
    void reset() override; // Logs if no specific reset
    bool loadInitialTensor(torch::Tensor &t) override;

    bool startHPC() override; // Logs if no HPC setup
    bool endHPC() override; // Logs if no HPC termination
    bool setConfig(INTELLI::ConfigMapPtr cfg) override; // Logs if no config setup
    bool setParams(CANDY::ParamPtr param) override; // Logs if no parameters set
    bool resetIndexStatistics() override; // Logs if no statistics reset

    INTELLI::ConfigMapPtr getIndexStatistics() override; // Logs if no statistics retrieval

    virtual bool insertTensor(const torch::Tensor &t) override = 0;

    virtual bool deleteTensor(torch::Tensor &t, int64_t k) override = 0;

    virtual bool reviseTensor(torch::Tensor &t, torch::Tensor &w) override = 0;

    virtual std::vector<torch::Tensor> searchTensor(const torch::Tensor &q, int64_t k) override = 0;
};
#endif // ANNS_ALGORITHM_BASE_HPP
