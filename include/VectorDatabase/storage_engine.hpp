#pragma once

#include <unordered_map>
#include <vector>
#include <string>
#include <fstream>

class StorageEngine {
public:
  StorageEngine(int dimensions, const std::string &path);

  void addVector(int id, const std::vector<float> &vec);
  void removeVector(int id);
  std::vector<float> getVector(int id) const;
  void persist();

private:
  int dimensions;
  std::string storagePath;
  std::unordered_map<int, std::vector<float>> data;

  void writeToDisk(int id, const std::vector<float> &vec);
  void deleteFromDisk(int id);
};
