//
// Created by zhonghao on 27/11/24.
//

#ifndef MANU_TIMETICK_HPP
#define MANU_TIMETICK_HPP

#include "HybridClock.hpp"

class TimeTick {
public:
  HybridClock timestamp;

  explicit TimeTick(const HybridClock& timestamp);
};

#endif // MANU_TIMETICK_HPP
