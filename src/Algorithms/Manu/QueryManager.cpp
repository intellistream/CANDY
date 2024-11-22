//
// Created by zhonghao on 22/11/24.
//

#include "Algorithms/Manu/QueryManager.hpp"

QueryManager::QueryManager(IndexManager* indexManager, StateManager* stateManager)
    : indexManager(indexManager), stateManager(stateManager) {}

std::vector<std::string> QueryManager::executeQuery(const std::string& queryParams) {
  uint64_t txnID = stateManager->startTransaction();

  // Validate the transaction's consistency.
  if (!stateManager->validateConsistency(0, 0, 1000)) {  // Example arguments.
    stateManager->rollbackTransaction(txnID);
    return {};
  }

  // Perform the query using IndexManager.
  auto results = indexManager->searchQuery(queryParams);

  // Commit the transaction.
  stateManager->commitTransaction(txnID);
  return results;
}

void QueryManager::coordinateSources() {
  // TODO: Ensure sources (WAL, BinLog, IndexManager) are synchronized.
}
