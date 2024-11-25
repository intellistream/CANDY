//
// Created by zhonghao on 22/11/24.
//
#include "Algorithms/Manu/IndexManager.hpp"
#include "Algorithms/Manu/DataStore.hpp"

IndexManager::IndexManager(WriteAheadLog* wal, BinLog* binLog, DataStore* dataStore)
    : wal(wal), binLog(binLog), dataStore(dataStore) {}

void IndexManager::addIndex(const std::string& indexType, const std::string& config) {
  dataStore->rebuildIndex(indexType); // Initialize an empty index for the type.
}

void IndexManager::updateIndex(const std::vector<std::string>& segmentData) {
  dataStore->rebuildIndex("default"); // Rebuild the "default" index after updates.
}

std::vector<std::string> IndexManager::searchQuery(const std::string& queryParams) {
  return dataStore->searchIndex("default", queryParams); // Use the "default" index for search.
}
