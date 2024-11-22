//
// Created by zhonghao on 22/11/24.
//
#include "Algorithms/Manu/IndexManager.hpp"

IndexManager::IndexManager(WriteAheadLog* wal, BinLog* binLog)
    : wal(wal), binLog(binLog) {}

void IndexManager::addIndex(const std::string& indexType, const std::string& config) {
  std::lock_guard<std::mutex> lock(indexMutex);
  indices[indexType] = {};  // Initialize an empty index for the given type.
}

void IndexManager::updateIndex(const std::vector<std::string>& segmentData) {
  std::lock_guard<std::mutex> lock(indexMutex);
  for (const auto& data : segmentData) {
    // TODO: Update indices with the provided segment data.
  }
}

std::vector<std::string> IndexManager::searchQuery(const std::string& queryParams) {
  std::lock_guard<std::mutex> lock(indexMutex);
  std::vector<std::string> results;

  // Query WAL for recent data.
  auto walResults = wal->scan(0, 100); // Example range; adjust as needed.
  for (const auto& entry : walResults) {
    results.push_back(entry.vectorData);  // Include data from WAL.
  }

  // Query BinLog for sealed batches.
  auto binLogBatch = binLog->readBatch(0);  // Example batch index.
  results.insert(results.end(), binLogBatch.begin(), binLogBatch.end());

  // TODO: Query other indices based on queryParams.
  return results;
}

