#pragma once

#include <VectorDatabase/indexing/anns.hpp>
#include <VectorDatabase/query_executor.hpp>
#include <VectorDatabase/storage_engine.hpp>
#include <string>
#include <vector>

class VectorDatabase {
public:
  // Constructor to initialize the database with dimensions and storage path
  VectorDatabase(int dimensions, const std::string &storagePath);

  // Add a vector to the database
  void addVector(int id, const std::vector<float> &vec);

  // Remove a vector by ID
  void removeVector(int id);

  // Perform a k-NN search
  std::vector<int> knnSearch(const std::vector<float> &query, int k);

  // Persist all data (manual trigger for persistence)
  void persist();

private:
  int dimensions;        // Dimensionality of the vectors
  StorageEngine storage; // Persistent storage engine
  ANNS index;            // Approximate Nearest Neighbor Search (ANNS) index
  QueryExecutor queryExecutor; // Query executor for vector operations
};
