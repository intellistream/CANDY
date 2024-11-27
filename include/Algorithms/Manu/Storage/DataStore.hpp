//
// Created by zhonghao on 27/11/24.
//

#ifndef MANU_DATASTORE_HPP
#define MANU_DATASTORE_HPP

#include "GrowingSegment.hpp"
#include "SealedSegment.hpp"
#include <unordered_map>
#include <string>

class Datastore {
private:
  std::unordered_map<std::string, GrowingSegment> growingSegments;
  std::unordered_map<std::string, SealedSegment> sealedSegments;

public:
  void insertIntoGrowingSegment(const std::string& segmentID, const std::vector<float>& vectorData);
  void sealSegment(const std::string& segmentID);
};

#endif // MANU_DATASTORE_HPP
