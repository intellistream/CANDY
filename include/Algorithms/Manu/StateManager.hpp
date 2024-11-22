//
// Created by zhonghao on 22/11/24.
// Manage multi-version concurrency control (MVCC), manage delta consistency by sync read and write to index
//

#ifndef STATE_MANAGER_HPP
#define STATE_MANAGER_HPP

#include <mutex>
#include <map>

class StateManager {
private:
  std::map<uint64_t, bool> activeTransactions;
  std::mutex txnMutex;

public:
  uint64_t startTransaction();
  void commitTransaction(uint64_t txnID);
  void rollbackTransaction(uint64_t txnID);
  bool validateConsistency(uint64_t lastUpdateTime, uint64_t requestTime, uint64_t stalenessTolerance);
};

#endif

