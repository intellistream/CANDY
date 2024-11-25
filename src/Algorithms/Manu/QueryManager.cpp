//
// Created by zhonghao on 22/11/24.
//

#include "Algorithms/Manu/QueryManager.hpp"

QueryManager::QueryManager(IndexManager* indexManager, StateManager* stateManager)
    : indexManager(indexManager), stateManager(stateManager) {}

std::vector<std::string> QueryManager::executeQuery(const std::string& queryParams) {
  uint64_t txnID = stateManager->startTransaction();

  if (!stateManager->validateConsistency(0, 0, 1000)) {
    stateManager->rollbackTransaction(txnID);
    return {};
  }

  auto results = indexManager->searchQuery(queryParams);

  stateManager->commitTransaction(txnID);
  return results;
}

void QueryManager::coordinateSources() {
  // TODO: Synchronize updates between WAL, BinLog, and IndexManager.
}
