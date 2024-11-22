//
// Created by zhonghao on 22/11/24.
// Log to store batch of index updates
//

#ifndef BIN_LOG_HPP
#define BIN_LOG_HPP

#include <vector>
#include <string>
#include <map>
#include <mutex>

class BinLog {
private:
  std::vector<std::string> batches;
  std::mutex binLogMutex;

public:
  void appendBatch(const std::vector<std::string>& batchData);
  std::vector<std::string> readBatch(uint64_t batchIndex);
};

#endif

