/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/9
 * Description: [Provide description here]
 */
#include "../../include/Core/vector_db.hpp"
#include <cmath>        // For distance calculations
#include <limits>       // For handling min distance comparison
#include <iostream>     // For logging

// Constructor
VectorDB::VectorDB() {
  // Initialize anything if needed
}

// Destructor
VectorDB::~VectorDB() {
  // Clean up any resources if needed
}

// Insert a new vector into the database (generates a unique ID for internal storage)
bool VectorDB::insert_vector(const std::vector<float>& vec) {
  std::unique_lock<std::shared_mutex> lock(db_mutex);  // Exclusive lock for writing
  size_t id = generate_id();
  vector_store[id] = vec;
  return true;
}

// Query the nearest vector based on Euclidean distance
std::optional<std::vector<float>> VectorDB::query_nearest_vector(const std::vector<float>& query_vec) const {
  std::shared_lock<std::shared_mutex> lock(db_mutex);  // Shared lock for reading

  if (vector_store.empty()) {
    std::cerr << "Database is empty. No vectors to search." << std::endl;
    return std::nullopt;
  }

  float min_distance = std::numeric_limits<float>::max();
  std::vector<float> nearest_vec;

  for (const auto& [id, vec] : vector_store) {
    float distance = calculate_distance(query_vec, vec);
    if (distance < min_distance) {
      min_distance = distance;
      nearest_vec = vec;
    }
  }

  return nearest_vec;
}

// Delete a vector by reference (removes exact matches)
bool VectorDB::delete_vector(const std::vector<float>& vec) {
  std::unique_lock<std::shared_mutex> lock(db_mutex);  // Exclusive lock for writing

  for (auto it = vector_store.begin(); it != vector_store.end(); ++it) {
    if (it->second == vec) {
      vector_store.erase(it);
      return true;
    }
  }

  std::cerr << "Vector not found for deletion!" << std::endl;
  return false;
}

// Get the total number of vectors in the database
size_t VectorDB::get_vector_count() const {
  std::shared_lock<std::shared_mutex> lock(db_mutex);  // Shared lock for reading
  return vector_store.size();
}

// Calculate Euclidean distance between two vectors
float VectorDB::calculate_distance(const std::vector<float>& vec1, const std::vector<float>& vec2) const {
  if (vec1.size() != vec2.size()) {
    throw std::invalid_argument("Vector dimensions do not match!");
  }

  float sum = 0.0;
  for (size_t i = 0; i < vec1.size(); ++i) {
    sum += std::pow(vec1[i] - vec2[i], 2);
  }
  return std::sqrt(sum);
}