#include <VectorDatabase/vector_database.hpp>
#include <iostream>
#include <stdexcept>

// Constructor to initialize storage, index, and query executor
VectorDatabase::VectorDatabase(int dimensions, const std::string &storagePath)
    : dimensions(dimensions), storage(dimensions, storagePath), index(dimensions), queryExecutor(&storage, &index) {}

// Add a vector to the database
void VectorDatabase::addVector(int id, const std::vector<float> &vec) {
  if (vec.size() != dimensions) {
    throw std::runtime_error("Vector dimensionality mismatch.");
  }

  // Add to storage and index
  storage.addVector(id, vec);
  index.addVector(id, vec);
}

// Remove a vector by ID
void VectorDatabase::removeVector(int id) {
  // Remove from storage and index
  storage.removeVector(id);
  index.removeVector(id);
}

// Perform a k-NN search
std::vector<int> VectorDatabase::knnSearch(const std::vector<float> &query, int k) {
  if (query.size() != dimensions) {
    throw std::runtime_error("Query vector dimensionality mismatch.");
  }

  return queryExecutor.knnSearch(query, k);
}

// Persist all data manually
void VectorDatabase::persist() {
  storage.persist();
}
