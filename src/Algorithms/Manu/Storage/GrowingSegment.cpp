//
// Created by zhonghao on 27/11/24.
//
#include "Algorithms/Manu/Storage/GrowingSegment.hpp"

void GrowingSegment::insertVector(const std::vector<float>& vectorData) {
  vectors.push_back(vectorData);
}

bool GrowingSegment::isThresholdReached() const {
  // TODO: Check if segment size threshold is reached
  return false;
}
