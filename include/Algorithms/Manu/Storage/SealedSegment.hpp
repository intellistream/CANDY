//
// Created by zhonghao on 27/11/24.
//

#ifndef MANU_SEALEDSEGMENT_HPP
#define MANU_SEALEDSEGMENT_HPP

#include <vector>

class SealedSegment {
private:
  std::vector<std::vector<float>> vectors;

public:
  explicit SealedSegment(const std::vector<std::vector<float>>& data);
};

#endif // MANU_SEALEDSEGMENT_HPP

