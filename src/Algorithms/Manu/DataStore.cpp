//
// Created by zhonghao on 25/11/24.
//
#include "Algorithms/Manu/DataStore.hpp"


void DataStore::addVector(const std::string& vectorData) {
  std::lock_guard<std::mutex> lock(storeMutex);
  vectors.push_back(vectorData);
}

std::vector<std::string> DataStore::getAllVectors() {
  std::lock_guard<std::mutex> lock(storeMutex);
  return vectors; // Return a copy of the vector list for thread safety.
}

void DataStore::rebuildIndex(const std::string& indexType) {
  std::lock_guard<std::mutex> lock(storeMutex);
  // Simulate rebuilding index with current vectors.
  indices[indexType] = vectors; // Use all vectors for index rebuild.
}

std::vector<std::string> DataStore::searchIndex(const std::string& indexType, const std::string& queryParams) {
  std::lock_guard<std::mutex> lock(storeMutex);
  auto it = indices.find(indexType);
  if (it != indices.end()) {
    return it->second; // Return indexed vectors.
  }
  return {};
}


