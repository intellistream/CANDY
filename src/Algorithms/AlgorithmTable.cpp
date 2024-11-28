/*
* Copyright (C) 2024 by the INTELLI team
* Created on: 2024/11/12
* Description: [Provide description here]
*/
#include <Algorithms/AlgorithmTable.hpp>
#include <Algorithms/FlatGPUIndex/FlatGPUIndex.hpp>
#include <Algorithms/HNSW/hnsw.hpp>
#include <Algorithms/KDTree/KDTree.hpp>
#include <Algorithms/KNN/KNNSearch.hpp>
#include <Algorithms/LSH/LSHSearch.hpp>

namespace CANDY_ALGO {
AlgorithmTable::AlgorithmTable() {
  indexMap["KNN"] = std::make_shared<KnnSearch>();
  //indexMap["KDTree"] = std::make_shared<KDTree>();
  indexMap["HNSW"] = std::make_shared<HNSW>();
  indexMap["FlatGPU"] = std::make_shared<FlatGPUIndex>();
  indexMap["LSH"] = std::make_shared<LSHSearch>();
  indexMap["Vamana"] = std::make_shared<Vamana>();
}
}  // namespace CANDY_ALGO