//
// Created by zhonghao on 22/11/24.
//
#include "Algorithms/Manu/BinLog.hpp"

void BinLog::appendBatch(const std::vector<std::string>& batchData) {
  std::lock_guard<std::mutex> lock(binLogMutex);
  // TODO: Serialize and store batchData into binlog.
}

std::vector<std::string> BinLog::readBatch(uint64_t batchIndex) {
  std::lock_guard<std::mutex> lock(binLogMutex);
  // TODO: Return the batch data for the given batchIndex.
  return {};
}
