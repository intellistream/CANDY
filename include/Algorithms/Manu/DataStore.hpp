//
// Created by zhonghao on 25/11/24.
//


#ifndef DATASTORE_HPP
#define DATASTORE_HPP

#include <vector>
#include <string>
#include <map>
#include <mutex>

class DataStore {
private:
  std::vector<std::string> vectors; // Vector data stored as serialized strings.
  std::map<std::string, std::vector<std::string>> indices; // Indices mapped by index type.
  std::mutex storeMutex; // Mutex for thread-safe access.

public:
  void addVector(const std::string& vectorData); // Add a vector to the datastore.
  std::vector<std::string> getAllVectors(); // Retrieve all stored vectors.

  void rebuildIndex(const std::string& indexType); // Rebuild the index for the given type.
  std::vector<std::string> searchIndex(const std::string& indexType, const std::string& queryParams); // Search the index for the given query.
};

#endif
