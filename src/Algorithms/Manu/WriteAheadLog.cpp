//
// Created by zhonghao on 22/11/24.
//

#include "Algorithms/Manu/WriteAheadLog.hpp"

WriteAheadLog::WriteAheadLog() : nextLSN(0) {}

uint64_t WriteAheadLog::appendEntry(const std::string& vectorData) {
  std::lock_guard<std::mutex> lock(logMutex);
  uint64_t lsn = nextLSN++;
  log.push_back({lsn, vectorData});
  return lsn;
}

void WriteAheadLog::flushSegment() {
  // TODO: Implement periodic flush to persistent storage.
}

std::vector<WriteAheadLog::LogEntry> WriteAheadLog::scan(uint64_t startLSN, uint64_t endLSN) {
  std::lock_guard<std::mutex> lock(logMutex);
  // TODO: Return log entries within the range [startLSN, endLSN].
  return {};
}
