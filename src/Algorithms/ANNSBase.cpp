/*
 * Copyright (C) 2024 by the INTELLI team
 * Created on: 2024/10/26 10:48
 * Description: ${DESCRIPTION}
 */

#include <Algorithms/ANNSBase.hpp>
#include <iostream>

namespace CANDY_ALGO {
// Reset the index to an initialized state
void ANNSBase::reset() {
  INTELLI_INFO("No specific reset implementation provided.");
}

bool ANNSBase::loadInitialTensor(torch::Tensor& t) {
  INTELLI_INFO(
      "No specific loadInitialTensor implementation provided. Just call the "
      "insertTensor function");
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

bool ANNSBase::insertTensor(const torch::Tensor& t) {
  assert(t.size(1));
  return false;
}

bool ANNSBase::reviseTensor(torch::Tensor& t, torch::Tensor& w) {
  assert(t.size(1) == w.size(1));
  return false;
}

bool ANNSBase::deleteTensor(torch::Tensor& t, int64_t k) {
  assert(t.size(1));
  assert(k > 0);
  return false;
}

std::vector<torch::Tensor> ANNSBase::searchTensor(const torch::Tensor& q,
                                                  int64_t k) {
  assert(k > 0);
  assert(q.size(1));
  std::vector<torch::Tensor> ru(1);
  ru[0] = torch::rand({1, 1});
  return ru;
}
}  // namespace CANDY_ALGO
