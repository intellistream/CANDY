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

class ANNSBase : public CANDY::AbstractANNS {
public:
    virtual ~ANNSBase() = default;

    virtual void reset() override;

    virtual bool setConfig(ConfigParserPtr cfg) override;

    virtual bool startHPC() override;

    virtual bool endHPC() override;

    virtual bool insertTensor(torch::Tensor &t) override;

    virtual bool loadInitialTensor(torch::Tensor &t) override;

    virtual bool deleteTensor(torch::Tensor &t, int64_t k = 1) override;

    virtual bool reviseTensor(torch::Tensor &t, torch::Tensor &w) override;

    virtual bool resetIndexStatistics() override;

    virtual ConfigParserPtr getIndexStatistics() override;
};

std::vector<std::string> u64ObjectToStringObject(std::vector<uint64_t> &u64s);

#endif // ANNS_ALGORITHM_BASE_HPP
