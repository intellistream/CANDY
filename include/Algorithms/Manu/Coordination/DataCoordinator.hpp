//
// Created by zhonghao on 27/11/24.
//

#ifndef MANU_DATACOORDINATOR_HPP
#define MANU_DATACOORDINATOR_HPP

#include <unordered_map>
#include <string>

class DataCoordinator {
private:
  std::unordered_map<std::string, std::string> segmentRoutes; // Maps segment ID to storage routes
public:
  void addSegmentRoute(const std::string& segmentID, const std::string& route);
  std::string getSegmentRoute(const std::string& segmentID) const;
};

#endif // MANU_DATACOORDINATOR_HPP
