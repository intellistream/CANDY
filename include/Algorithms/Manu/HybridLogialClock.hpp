//
// Created by zhonghao on 22/11/24.
// A unique lock structure proposed by Manu, combining both query timestamp and system execution timestamp
//

#ifndef HYBRID_LOGICAL_CLOCK_HPP
#define HYBRID_LOGICAL_CLOCK_HPP

#include <atomic>
#include <mutex>
#include <chrono>

class HybridLogicalClock {
private:
  std::atomic<uint64_t> physicalTime;
  std::atomic<uint64_t> logicalCounter;

public:
  HybridLogicalClock();
  uint64_t generateTimestamp();
  void trackStaleness(uint64_t lastUpdateTime, uint64_t requestTime, uint64_t stalenessTolerance);
};

#endif

