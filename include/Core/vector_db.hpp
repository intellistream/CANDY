/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/9
 * Description: [Provide description here]
 */
#ifndef INTELLISTREAM_SRC_CORE_VECTOR_DB_HPP_
#define INTELLISTREAM_SRC_CORE_VECTOR_DB_HPP_

#ifndef VECTOR_DB_H
#define VECTOR_DB_H

#include <unordered_map>
#include <vector>
#include <shared_mutex>
#include <memory>

// Forward declaration of the base search algorithm interface
class SearchAlgorithm;

class VectorDB {
 public:
  // Constructor and Destructor
  VectorDB(size_t dimensions, std::shared_ptr<SearchAlgorithm> search_algorithm);
  ~VectorDB();

  // Insert a vector into the database
  bool insert_vector(const std::vector<float> &vec);

  // Query the nearest vector(s) using the assigned search algorithm
  std::vector<std::vector<float>> query_nearest_vectors(const std::vector<float> &query_vec, size_t k = 1) const;

  // Delete a vector by reference (removes exact matches)
  bool delete_vector(const std::vector<float> &vec);

  // Get the total number of vectors in the database
  size_t get_vector_count() const;

 private:
  // Internal storage for vectors (indexed by their ID)
  std::unordered_map<size_t, std::vector<float>> vector_store;

  // Mutex for thread safety
  mutable std::shared_mutex db_mutex;

  // Number of dimensions of the vectors
  size_t dimensions;

  // Search algorithm to use for querying (e.g., k-NN, Approximate NN)
  std::shared_ptr<SearchAlgorithm> search_algorithm;

  // ID generation for vectors
  size_t next_id = 0;
  size_t generate_id() { return next_id++; }
};

#endif // VECTOR_DB_H

#endif //INTELLISTREAM_SRC_CORE_VECTOR_DB_HPP_
