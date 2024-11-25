//
// Created by zhonghao on 22/11/24.
// Manages multiple indexing algorithms under Manu concurrency framework
//

#ifndef INDEX_MANAGER_HPP
#define INDEX_MANAGER_HPP

#include <vector>
#include <string>
#include <map>
#include <mutex>
#include "WriteAheadLog.hpp"
#include "BinLog.hpp"
#include "DataStore.hpp"

class IndexManager {
private:
  WriteAheadLog* wal;
  BinLog* binLog;
  DataStore* dataStore;
  std::mutex indexMutex;

public:
  IndexManager(WriteAheadLog* wal, BinLog* binLog, DataStore* dataStore);

  void addIndex(const std::string& indexType, const std::string& config);
  void updateIndex(const std::vector<std::string>& segmentData);
  std::vector<std::string> searchQuery(const std::string& queryParams);
};

#endif

