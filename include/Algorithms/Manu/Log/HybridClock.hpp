//
// Created by zhonghao on 27/11/24.
//

#ifndef MANU_HYBRIDCLOCK_HPP
#define MANU_HYBRIDCLOCK_HPP

#include <chrono>

class HybridClock {
public:
  std::chrono::system_clock::time_point realTime;
  unsigned long logicalClock;

  HybridClock();
  void updateLogicalClock(unsigned long newLogicalClock);
};

#endif // MANU_HYBRIDCLOCK_HPP
