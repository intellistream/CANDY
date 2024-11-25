//
// Created by zhonghao on 22/11/24.
//
#include "Algorithms/Manu/BinLog.hpp"


void BinLog::appendBatch(const std::vector<std::string>& batchData) {
  std::lock_guard<std::mutex> lock(binLogMutex);
  batches.insert(batches.end(), batchData.begin(), batchData.end());
}

std::vector<std::string> BinLog::readBatch(uint64_t batchIndex) {
  std::lock_guard<std::mutex> lock(binLogMutex);
  if (batchIndex < batches.size()) {
    return {batches[batchIndex]};
  }
  return {};
}
