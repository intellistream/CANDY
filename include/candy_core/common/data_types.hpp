#ifndef CANDY_COMMON_DATA_TYPES_HPP
#define CANDY_COMMON_DATA_TYPES_HPP

#include <vector>
#include <string>
#include <memory>
#include <functional>

namespace candy {

// Common data type for vector data
using VectorData = std::vector<float>;

// Wrapper for vector data with metadata (e.g., ID, timestamp)
struct VectorRecord {
  const std::string id;                      // Unique identifier for the vector
  std::shared_ptr<VectorData> data;         // The vector data itself
  const int64_t timestamp;                  // Timestamp for the record

  // Constructor with move semantics for efficiency
  VectorRecord(std::string id, VectorData data, int64_t timestamp)
      : id(std::move(id)),
        data(std::make_shared<VectorData>(std::move(data))),
        timestamp(timestamp) {}

  // Default constructor (optional)
  VectorRecord() : id(""), data(std::make_shared<VectorData>()), timestamp(0) {}

  // Equality operator for comparisons
  bool operator==(const VectorRecord &other) const {
    return id == other.id && timestamp == other.timestamp &&
           *data == *other.data;
  }
};

// Hash function for unordered containers (optional)
struct VectorRecordHash {
  std::size_t operator()(const VectorRecord &record) const {
    return std::hash<std::string>()(record.id) ^ std::hash<int64_t>()(record.timestamp);
  }
};

// Helper function to create a vector record
inline std::shared_ptr<VectorRecord> create_vector_record(const std::string &id, const VectorData &data, int64_t timestamp) {
  return std::make_shared<VectorRecord>(id, data, timestamp);
}

} // namespace candy

#endif // CANDY_COMMON_DATA_TYPES_HPP
