//
// Created by zhonghao on 27/11/24.
//

#ifndef MANU_WAL_HPP
#define MANU_WAL_HPP

#include <string>
#include <vector>

class WAL {
public:
  enum class Type { INSERTION, SEARCH };

  Type walType;
  std::string vectorID;
  std::vector<float> vectorData;
  std::string label;
  float numericalField;

  WAL(Type type, const std::string& vectorID, const std::vector<float>& vectorData,
      const std::string& label, float numericalField);
};

#endif // MANU_WAL_HPP
