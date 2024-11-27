//
// Created by zhonghao on 27/11/24.
//
#include "Algorithms/Manu/Coordination/DataCoordinator.hpp"

void DataCoordinator::addSegmentRoute(const std::string& segmentID, const std::string& route) {
  segmentRoutes[segmentID] = route;
}

std::string DataCoordinator::getSegmentRoute(const std::string& segmentID) const {
  auto it = segmentRoutes.find(segmentID);
  if (it != segmentRoutes.end()) {
    return it->second;
  }
  return {};
}
