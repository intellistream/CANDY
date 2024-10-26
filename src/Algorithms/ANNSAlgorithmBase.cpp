/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: shuhao
 * Created on: 2024/10/26 10:48
 * Modified by:
 * Modified on:
 * Description: ${DESCRIPTION}
 */

#include <Algorithms/ANNSAlgorithmBase.hpp>
#include <cassert>
#include <iostream>
#include <cstring>

// Definition of utility function
std::vector<std::string> u64ObjectToStringObject(std::vector<uint64_t>& u64s) {
    std::vector<std::string> ru(u64s.size());
    for (size_t i = 0; i < u64s.size(); i++) {
        uint64_t u64i = u64s[i];
        const char* char_ptr = reinterpret_cast<const char*>(&u64i);
        ru[i] = std::string(char_ptr, sizeof(uint64_t));
    }
    return ru;
}

void ANNSAlgorithmBase::reset() {
    // Default implementation for reset
    std::cout << "Resetting index to initial state." << std::endl;
}

bool ANNSAlgorithmBase::setConfigClass(const ConfigParser& cfg) {
    ConfigParserPtr cfgPtr = std::make_shared<ConfigParser>(cfg);
    return setConfig(cfgPtr);
}

bool ANNSAlgorithmBase::setConfig(const ConfigParserPtr& cfg) {
    assert(cfg);
    std::string metricType = cfg->get_string("metricType", "L2");
    metric = METRIC_L2;
    if (metricType == "dot" || metricType == "IP" || metricType == "cossim") {
        metric = METRIC_INNER_PRODUCT;
    }
    std::cout << "Configuration set with metric type: " << metricType << std::endl;
    return true;
}

bool ANNSAlgorithmBase::setFrozenLevel(int64_t frozenLv) {
    assert(frozenLv >= 0);
    std::cout << "Setting frozen level to: " << frozenLv << std::endl;
    return true;
}

bool ANNSAlgorithmBase::insertTensor(const torch::Tensor& t) {
    assert(t.size(1) > 0);
    std::cout << "Inserting tensor with size: " << t.sizes() << std::endl;
    return true;
}

bool ANNSAlgorithmBase::loadInitialTensor(const torch::Tensor& t) {
    std::cout << "Loading initial tensor with size: " << t.sizes() << std::endl;
    return insertTensor(t);
}

bool ANNSAlgorithmBase::deleteTensor(const torch::Tensor& t, int64_t k) {
    assert(t.size(1) > 0);
    assert(k > 0);
    std::cout << "Deleting tensor with size: " << t.sizes() << " and k: " << k << std::endl;
    return true;
}

bool ANNSAlgorithmBase::reviseTensor(const torch::Tensor& t, const torch::Tensor& w) {
    assert(t.size(1) == w.size(1));
    std::cout << "Revising tensor with sizes: " << t.sizes() << " and " << w.sizes() << std::endl;
    return true;
}

bool ANNSAlgorithmBase::startHPC() {
    isHPCStarted = true;
    std::cout << "Starting HPC features." << std::endl;
    return true;
}

bool ANNSAlgorithmBase::endHPC() {
    isHPCStarted = false;
    std::cout << "Ending HPC features." << std::endl;
    return true;
}

bool ANNSAlgorithmBase::offlineBuild(const torch::Tensor& t) {
    std::cout << "Performing offline build with tensor size: " << t.sizes() << std::endl;
    return true;
}

bool ANNSAlgorithmBase::waitPendingOperations() {
    std::cout << "Waiting for all pending operations to complete." << std::endl;
    return true;
}

bool ANNSAlgorithmBase::loadInitialStringObject(const torch::Tensor& t, const std::vector<std::string>& strs) {
    assert(t.size(1) > 0);
    assert(strs.size() > 0);
    std::cout << "Loading initial string objects with tensor size: " << t.sizes() << std::endl;
    return true;
}

bool ANNSAlgorithmBase::insertStringObject(const torch::Tensor& t, const std::vector<std::string>& strs) {
    assert(t.size(1) > 0);
    assert(strs.size() > 0);
    std::cout << "Inserting string objects with tensor size: " << t.sizes() << std::endl;
    return true;
}

bool ANNSAlgorithmBase::deleteStringObject(const torch::Tensor& t, int64_t k) {
    assert(t.size(1) > 0);
    assert(k > 0);
    std::cout << "Deleting string objects with tensor size: " << t.sizes() << " and k: " << k << std::endl;
    return true;
}

bool ANNSAlgorithmBase::deleteU64Object(const torch::Tensor& t, int64_t k) {
    return deleteStringObject(t, k);
}

bool ANNSAlgorithmBase::loadInitialTensorAndQueryDistribution(const torch::Tensor& t, const torch::Tensor& query) {
    assert(query.size(0) > 0);
    std::cout << "Loading initial tensor and query distribution." << std::endl;
    return loadInitialTensor(t);
}

bool ANNSAlgorithmBase::resetIndexStatistics() {
    std::cout << "Resetting index statistics." << std::endl;
    return true;
}

ConfigParserPtr ANNSAlgorithmBase::getIndexStatistics() {
    auto ru = std::make_shared<ConfigParser>();
    ru->edit("hasExtraStatistics", 0);
    std::cout << "Getting index statistics." << std::endl;
    return ru;
}
