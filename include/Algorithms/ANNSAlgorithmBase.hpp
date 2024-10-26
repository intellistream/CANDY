/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: shuhao
 * Created on: 2024/10/26 10:48
 * Modified by:
 * Modified on:
 * Description: ${DESCRIPTION}
 */

// ANNSAlgorithmBase.hpp
#ifndef ANNS_ALGORITHM_BASE_HPP
#define ANNS_ALGORITHM_BASE_HPP

#include <torch/torch.h>
#include <vector>
#include <string>
#include <Algorithms/AbstractANNSAlgorithm.hpp>

class ANNSAlgorithmBase : public AbstractANNSAlgorithm {
public:
    virtual ~ANNSAlgorithmBase() = default;

    // Methods with default implementation in ANNSAlgorithmBase
    virtual void reset() override;

    virtual bool setConfigClass(const ConfigParser &cfg) override;

    virtual bool setConfig(const ConfigParserPtr &cfg) override;

    virtual bool setFrozenLevel(int64_t frozenLv) override;

    virtual bool insertTensor(const torch::Tensor &t) override;

    virtual bool loadInitialTensor(const torch::Tensor &t) override;

    virtual bool deleteTensor(const torch::Tensor &t, int64_t k = 1) override;

    virtual bool reviseTensor(const torch::Tensor &t, const torch::Tensor &w) override;

    virtual bool startHPC() override;

    virtual bool endHPC() override;

    virtual bool offlineBuild(const torch::Tensor &t) override;

    virtual bool waitPendingOperations() override;

    virtual bool loadInitialStringObject(const torch::Tensor &t, const std::vector<std::string> &strs) override;

    virtual bool insertStringObject(const torch::Tensor &t, const std::vector<std::string> &strs) override;

    virtual bool deleteStringObject(const torch::Tensor &t, int64_t k = 1) override;

    virtual bool deleteU64Object(const torch::Tensor &t, int64_t k = 1) override;

    virtual bool loadInitialTensorAndQueryDistribution(const torch::Tensor &t, const torch::Tensor &query) override;

    virtual bool resetIndexStatistics() override;

    virtual ConfigParserPtr getIndexStatistics() override;

    // Pure virtual methods to be implemented by derived classes
    virtual std::vector<idx_t> searchIndex(const torch::Tensor &q, int64_t k) override = 0;

    virtual std::vector<torch::Tensor> getTensorByIndex(const std::vector<idx_t> &idx, int64_t k) override = 0;

    virtual torch::Tensor rawData() override = 0;

    virtual std::vector<torch::Tensor> searchTensor(const torch::Tensor &q, int64_t k) override = 0;

    virtual std::vector<std::vector<std::string> > searchStringObject(const torch::Tensor &q, int64_t k) override = 0;

    virtual std::vector<std::vector<uint64_t> > searchU64Object(const torch::Tensor &q, int64_t k) override = 0;

    virtual std::tuple<std::vector<torch::Tensor>, std::vector<std::vector<std::string> > > searchTensorAndStringObject(
        const torch::Tensor &q, int64_t k) override = 0;
};

std::vector<std::string> u64ObjectToStringObject(std::vector<uint64_t> &u64s);

#endif // ANNS_ALGORITHM_BASE_HPP
