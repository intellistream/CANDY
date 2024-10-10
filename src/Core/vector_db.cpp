/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/9
 * Description: [Provide description here]
 */
#include <Core/vector_db.hpp>
#include <Algorithms/search_algorithm.hpp>  // Base search algorithm interface

#include <iostream>
#include <mutex>

// Constructor with dimensions and search algorithm injection
VectorDB::VectorDB(size_t dimensions, std::shared_ptr<SearchAlgorithm> search_algorithm)
    : dimensions(dimensions), search_algorithm(search_algorithm) {}

// Destructor
VectorDB::~VectorDB() {
  // Clean up if necessary
}

// Insert a vector into the database
bool VectorDB::insert_vector(const std::vector<float> &vec) {
  if (vec.size() != dimensions) {
    std::cerr << "Error: Vector dimensions do not match the expected size." << std::endl;
    return false;
  }

  std::unique_lock<std::shared_mutex> lock(db_mutex);

  size_t id = generate_id();
  vector_store[id] = vec;

  // Insert the vector into the search algorithm's index
  search_algorithm->insert(id, vec);

  return true;
}

// Query the nearest vectors using the assigned search algorithm
std::vector<std::vector<float>> VectorDB::query_nearest_vectors(const std::vector<float> &query_vec, size_t k) const {
  if (query_vec.size() != dimensions) {
    std::cerr << "Error: Query vector dimensions do not match." << std::endl;
    return {};
  }

  std::shared_lock<std::shared_mutex> lock(db_mutex);

  // Use the search algorithm to retrieve the nearest neighbors
  std::vector<size_t> nearest_ids = search_algorithm->query(query_vec, k);

  // Convert the IDs back to vectors
  std::vector<std::vector<float>> nearest_vectors;
  for (size_t id : nearest_ids) {
    nearest_vectors.push_back(vector_store.at(id));
  }

  return nearest_vectors;
}

// Delete a vector from the database
bool VectorDB::delete_vector(const std::vector<float> &vec) {
  std::unique_lock<std::shared_mutex> lock(db_mutex);

  for (auto it = vector_store.begin(); it != vector_store.end(); ++it) {
    if (it->second == vec) {
      size_t id = it->first;

      // Remove from the search algorithm's index
      search_algorithm->remove(id);

      // Remove from the vector store
      vector_store.erase(it);

      return true;
    }
  }

  std::cerr << "Vector not found for deletion!" << std::endl;
  return false;
}

// Get the total number of vectors in the database
size_t VectorDB::get_vector_count() const {
  std::shared_lock<std::shared_mutex> lock(db_mutex);
  return vector_store.size();
}