//
// Created by zhonghao on 22/11/24.
// Log to store streaming index updates.
//

#ifndef WRITE_AHEAD_LOG_HPP
#define WRITE_AHEAD_LOG_HPP

#include <vector>
#include <mutex>
#include <string>
#include <map>

class WriteAheadLog {
private:
  struct LogEntry {
    uint64_t lsn;
    std::string vectorData; // Serialized vector data
  };

  std::vector<LogEntry> log;
  std::mutex logMutex;
  uint64_t nextLSN;

public:
  WriteAheadLog();
  uint64_t appendEntry(const std::string& vectorData);
  void flushSegment();
  std::vector<LogEntry> scan(uint64_t startLSN, uint64_t endLSN);
};

#endif

