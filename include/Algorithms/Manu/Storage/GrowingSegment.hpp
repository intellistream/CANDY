//
// Created by zhonghao on 27/11/24.
//

#ifndef MANU_GROWINGSEGMENT_HPP
#define MANU_GROWINGSEGMENT_HPP

#include <vector>

class GrowingSegment {
private:
  std::vector<std::vector<float>> vectors;

public:
  void insertVector(const std::vector<float>& vectorData);
  bool isThresholdReached() const;
};

#endif // MANU_GROWINGSEGMENT_HPP

