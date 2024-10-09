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
#include <string>
#include <shared_mutex>  // For std::shared_mutex
#include <mutex>         // For std::unique_lock
#include <optional>      // For std::optional (C++17)

class VectorDB {
 public:
  // Constructor and Destructor
  VectorDB();
  ~VectorDB();

  // Insert a vector into the database
  bool insert_vector(const std::vector<float>& vec);

  // Find the nearest vector based on a query vector (returns the closest match)
  std::optional<std::vector<float>> query_nearest_vector(const std::vector<float>& query_vec) const;

  // Delete a vector by reference (removes exact matches)
  bool delete_vector(const std::vector<float>& vec);

  // Get the total number of vectors in the database
  size_t get_vector_count() const;

 private:
  // Helper function to calculate Euclidean distance
  float calculate_distance(const std::vector<float>& vec1, const std::vector<float>& vec2) const;

  // Internal storage for vectors (keyed by an ID for internal purposes, optional)
  std::unordered_map<size_t, std::vector<float>> vector_store;

  // Mutex for thread safety
  mutable std::shared_mutex db_mutex;

  // Simple hash function to assign an ID (or could use a hash of the vector itself)
  size_t next_id = 0;
  size_t generate_id() { return next_id++; }
};

#endif // VECTOR_DB_H


#endif //INTELLISTREAM_SRC_CORE_VECTOR_DB_HPP_
