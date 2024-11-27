//
// Created by zhonghao on 27/11/24.
//
#include "Algorithms/Manu/Log/HybridClock.hpp"

HybridClock::HybridClock() : realTime(std::chrono::system_clock::now()), logicalClock(0) {}

void HybridClock::updateLogicalClock(unsigned long newLogicalClock) {
  logicalClock = newLogicalClock;
}
