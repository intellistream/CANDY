/*
 * Copyright (C) 2024 by the INTELLI team
 * Created on: 2024/10/26 10:48
 * Description: ${DESCRIPTION}
 */

#include <Algorithms/ANNSBase.hpp>
#include <iostream>

// Reset the index to an initialized state
void ANNSBase::reset() {
    INTELLI_INFO("No specific reset implementation provided.");
}

bool ANNSBase::loadInitialTensor(torch::Tensor &t) {
    INTELLI_INFO("No specific loadInitialTensor implementation provided. Just call the insertTensor function");
    return insertTensor(t);
}

// Start the High-Performance Computation (HPC) setup
bool ANNSBase::startHPC() {
    INTELLI_INFO("No specific startHPC implementation provided.");
    return true;
}

// End the High-Performance Computation (HPC) process
bool ANNSBase::endHPC() {
    INTELLI_INFO("No specific endHPC implementation provided.");
    return true;
}

// Set configuration (with logging when no specific behavior is provided)
bool ANNSBase::setConfig(INTELLI::ConfigMapPtr cfg) {
    INTELLI_INFO("No specific setConfig implementation provided.");
    return true;
}

// Set parameters (with logging when no specific behavior is provided)
bool ANNSBase::setParams(CANDY::ParamPtr param) {
    INTELLI_INFO("No parameters to be set.");
    return true;
}

// Reset any collected index statistics
bool ANNSBase::resetIndexStatistics() {
    INTELLI_INFO("No specific resetIndexStatistics implementation provided.");
    return true;
}

// Retrieve index statistics
INTELLI::ConfigMapPtr ANNSBase::getIndexStatistics() {
    INTELLI_INFO("No specific getIndexStatistics implementation provided.");
    return nullptr;
}
