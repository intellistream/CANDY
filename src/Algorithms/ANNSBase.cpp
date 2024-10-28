/*
 * Copyright (C) 2024 by the INTELLI team
 * Created on: 2024/10/26 10:48
 * Description: ${DESCRIPTION}
 */

#include <Algorithms/ANNSBase.hpp>
#include <iostream>

// Reset the index to an initialized state
void ANNSBase::reset() {
    INTELLI_INFO("No specific reset implementation provided in ANNSBase.");
}

// Start the High-Performance Computation (HPC) setup
bool ANNSBase::startHPC() {
    INTELLI_INFO("No specific startHPC implementation provided in ANNSBase.");
    return false;
}

// End the High-Performance Computation (HPC) process
bool ANNSBase::endHPC() {
    INTELLI_INFO("No specific endHPC implementation provided in ANNSBase.");
    return false;
}

// Set configuration (with logging when no specific behavior is provided)
bool ANNSBase::setConfig(INTELLI::ConfigMapPtr cfg) {
    INTELLI_INFO("No specific setConfig implementation provided in ANNSBase.");
    return false;
}

// Set parameters (with logging when no specific behavior is provided)
bool ANNSBase::setParams(CANDY::ParamPtr param) {
    INTELLI_INFO("No parameters to be set in ANNSBase.");
    return false;
}

// Reset any collected index statistics
bool ANNSBase::resetIndexStatistics() {
    INTELLI_INFO("No specific resetIndexStatistics implementation provided in ANNSBase.");
    return false;
}

// Retrieve index statistics
INTELLI::ConfigMapPtr ANNSBase::getIndexStatistics() {
    INTELLI_INFO("No specific getIndexStatistics implementation provided in ANNSBase.");
    return nullptr;
}
