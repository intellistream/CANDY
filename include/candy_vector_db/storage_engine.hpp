#ifndef CANDY_VECTOR_DB_STORAGE_ENGINE_HPP
#define CANDY_VECTOR_DB_STORAGE_ENGINE_HPP

#include <candy_core/common/data_types.hpp>
#include <fstream>
#include <string>
#include <unordered_map>

namespace candy {

class StorageEngine {
public:
  explicit StorageEngine(const std::string &storagePath);

  // Add a vector to storage
  void add(const std::shared_ptr<VectorRecord> &record);

  // Remove a vector from storage
  void remove(const std::string &id);

  // Load vectors from disk
  void load();

  // Persist all vectors to disk
  void persist();

private:
  std::unordered_map<std::string, std::shared_ptr<VectorRecord>> data;
  std::string storagePath;

  // Helper: Write a single vector to disk
  void writeToDisk(const std::string &id, const VectorData &vec);

  // Helper: Delete a vector from disk
  void deleteFromDisk(const std::string &id);
};

} // namespace candy

#endif // CANDY_VECTOR_DB_STORAGE_ENGINE_HPP
