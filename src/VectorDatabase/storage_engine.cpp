#include <VectorDatabase/storage_engine.hpp>
#include <stdexcept>

StorageEngine::StorageEngine(int dimensions, const std::string &path)
    : dimensions(dimensions), storagePath(path) {}

// Add a vector to storage
void StorageEngine::addVector(int id, const std::vector<float> &vec) {
  if (vec.size() != dimensions) {
    throw std::runtime_error("Vector dimensionality mismatch.");
  }
  data[id] = vec;
  writeToDisk(id, vec);
}

// Remove a vector from storage
void StorageEngine::removeVector(int id) {
  if (data.erase(id) == 0) {
    throw std::runtime_error("Vector ID not found.");
  }
  deleteFromDisk(id);
}

// Retrieve a vector by ID
std::vector<float> StorageEngine::getVector(int id) const {
  auto it = data.find(id);
  if (it == data.end()) {
    throw std::runtime_error("Vector ID not found.");
  }
  return it->second;
}

// Persist all vectors to disk
void StorageEngine::persist() {
  std::ofstream outFile(storagePath + "/vectors.dat");
  for (const auto &[id, vec] : data) {
    outFile << id;
    for (const auto &val : vec) {
      outFile << " " << val;
    }
    outFile << "\n";
  }
  outFile.close();
}

// Helper: Write a single vector to disk
void StorageEngine::writeToDisk(int id, const std::vector<float> &vec) {
  std::ofstream outFile(storagePath + "/vectors.dat", std::ios::app);
  outFile << id;
  for (const auto &val : vec) {
    outFile << " " << val;
  }
  outFile << "\n";
  outFile.close();
}

// Helper: Delete a vector from disk (placeholder for simplicity)
void StorageEngine::deleteFromDisk(int id) {
  // Implementation can involve marking the vector as deleted or rebuilding the storage file.
  // Placeholder: Not implemented in this example.
}
