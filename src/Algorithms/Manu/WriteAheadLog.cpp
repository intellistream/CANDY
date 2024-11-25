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
  // TODO: Integrate logic to persist log data to storage or binlog.
}

std::vector<WriteAheadLog::LogEntry> WriteAheadLog::scan(uint64_t startLSN, uint64_t endLSN) {
  std::lock_guard<std::mutex> lock(logMutex);
  std::vector<LogEntry> results;
  for (const auto& entry : log) {
    if (entry.lsn >= startLSN && entry.lsn < endLSN) {
      results.push_back(entry);
    }
  }
  return results;
}
