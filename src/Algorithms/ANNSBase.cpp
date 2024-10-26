/*
 * Copyright (C) 2024 by the INTELLI team
 * Created on: 2024/10/26 10:48
 * Description: ${DESCRIPTION}
 */

#include <Algorithms/ANNSBase.hpp>
#include <iostream>

void ANNSBase::reset() {
    // Default implementation for reset
    std::cout << "Resetting index to initial state." << std::endl;
}

bool ANNSBase::startHPC() {
    isHPCStarted = true;
    std::cout << "Starting HPC features." << std::endl;
    return true;
}

bool ANNSBase::endHPC() {
    isHPCStarted = false;
    std::cout << "Ending HPC features." << std::endl;
    return true;
}

bool ANNSBase::resetIndexStatistics() {
    std::cout << "Resetting index statistics." << std::endl;
    return true;
}

INTELLI::ConfigMapPtr ANNSBase::getIndexStatistics() {
    auto ru = std::make_shared<INTELLI::ConfigMap>();
    ru->edit("hasExtraStatistics", 0);
    std::cout << "Getting index statistics." << std::endl;
    return ru;
}
