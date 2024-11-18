/*
* Copyright (C) 2024 by the INTELLI team
* Created on: 2024/11/12
* Description: [Provide description here]
*/
#ifndef CANDY_ALGORITHMTABLE_H
#define CANDY_ALGORITHMTABLE_H
#include <map>
#include <string>
#include <Algorithms/ANNSBase.hpp>
namespace CANDY_ALGO {
class AlgorithmTable{
 protected:
  std::map<std::string, CANDY_ALGO::ANNSBasePtr> indexMap;
 public:
  AlgorithmTable();
  ~AlgorithmTable() {}
  /**
   * @brief To register a new ALGO
   * @param anew The new algo
   * @param tag THe name tag
   */
  void addAlgorithm(CANDY_ALGO::ANNSBasePtr anew, std::string tag) {
    indexMap[tag] = anew;
  }
  /**
   * @brief find an index in the table according to its name
   * @param name The nameTag of index
   * @return The ANNSBasePtr, nullptr if not found
   */
  CANDY_ALGO::ANNSBasePtr getIndex(std::string name) {
    if (indexMap.count(name)) {
      return indexMap[name];
    }
    return nullptr;
  }
};
};
#endif  //CANDY_ALGORITHMTABLE_H