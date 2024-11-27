//
// Created by zhonghao on 27/11/24.
//

#ifndef MANU_QUERYCOORDINATOR_HPP
#define MANU_QUERYCOORDINATOR_HPP

#include <unordered_map>
#include <string>
#include <vector>

class QueryCoordinator {
private:
  std::unordered_map<std::string, std::vector<std::string>> queryNodeToSegments; // Maps query node to its segments
public:
  void mapQueryNodeToSegments(const std::string& queryNodeID, const std::vector<std::string>& segments);
  std::vector<std::string> getSegmentsForQueryNode(const std::string& queryNodeID) const;
};

#endif // MANU_QUERYCOORDINATOR_HPP
