//
// Created by zhonghao on 27/11/24.
//
#include "Algorithms/Manu/Coordination/QueryCoordinator.hpp"

void QueryCoordinator::mapQueryNodeToSegments(const std::string& queryNodeID, const std::vector<std::string>& segments) {
  queryNodeToSegments[queryNodeID] = segments;
}

std::vector<std::string> QueryCoordinator::getSegmentsForQueryNode(const std::string& queryNodeID) const {
  auto it = queryNodeToSegments.find(queryNodeID);
  if (it != queryNodeToSegments.end()) {
    return it->second;
  }
  return {};
}
