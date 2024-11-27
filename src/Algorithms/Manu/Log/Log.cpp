//
// Created by zhonghao on 27/11/24.
//
#include "Algorithms/Manu/Log/Log.hpp"

Log::Log(const std::string& logID, const HybridClock& time)
    : uniqueLogID(logID), timestamp(time) {}
