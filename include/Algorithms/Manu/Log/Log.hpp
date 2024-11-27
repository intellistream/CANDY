//
// Created by zhonghao on 27/11/24.
//

#ifndef MANU_LOG_HPP
#define MANU_LOG_HPP

#include "HybridClock.hpp"
#include <string>

class Log {
public:
  std::string uniqueLogID;
  HybridClock timestamp;

  Log(const std::string& logID, const HybridClock& time);
};

#endif // MANU_LOG_HPP

