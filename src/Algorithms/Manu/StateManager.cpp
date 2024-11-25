//
// Created by zhonghao on 22/11/24.
//
#include "Algorithms/Manu/StateManager.hpp"

uint64_t StateManager::startTransaction() {
  std::lock_guard<std::mutex> lock(txnMutex);
  uint64_t txnID = activeTransactions.size() + 1;
  activeTransactions[txnID] = true;
  return txnID;
}

void StateManager::commitTransaction(uint64_t txnID) {
  std::lock_guard<std::mutex> lock(txnMutex);
  activeTransactions.erase(txnID);
}

void StateManager::rollbackTransaction(uint64_t txnID) {
  std::lock_guard<std::mutex> lock(txnMutex);
  activeTransactions.erase(txnID);
}

bool StateManager::validateConsistency(uint64_t lastUpdateTime, uint64_t requestTime, uint64_t stalenessTolerance) {
  return (requestTime - lastUpdateTime) >= stalenessTolerance;
}
