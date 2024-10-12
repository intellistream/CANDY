/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/9
 * Description: [Provide description here]
 */
#include <Algorithms/knn_search.hpp>
#include <algorithm>
#include <stdexcept>

// Constructor
KnnSearch::KnnSearch(size_t dimensions) : dimensions(dimensions) {}

// Insert a vector into the k-NN index
void KnnSearch::insert(size_t id, const std::vector<float> &vec) {
  if (vec.size() != dimensions) {
    throw std::invalid_argument("Vector dimensions do not match");
  }
  index[id] = vec;
}

// Query k nearest neighbors
std::vector<size_t> KnnSearch::query(const std::vector<float> &query_vec, size_t k) const {
  if (query_vec.size() != dimensions) {
    throw std::invalid_argument("Query vector dimensions do not match");
  }

  // Brute-force search: calculate distance to all stored vectors
  std::vector<std::pair<float, size_t>> distances;  // Pair of distance and ID
  for (const auto &[id, vec] : index) {
    float dist = calculate_distance(query_vec, vec);
    distances.emplace_back(dist, id);
  }

  // Sort by distance
  std::sort(distances.begin(), distances.end());

  // Return the top k nearest neighbors' IDs
  std::vector<size_t> nearest_ids;
  for (size_t i = 0; i < std::min(k, distances.size()); ++i) {
    nearest_ids.push_back(distances[i].second);
  }

  return nearest_ids;
}

// Remove a vector from the k-NN index
void KnnSearch::remove(size_t id) {
  index.erase(id);
}

// Calculate Euclidean distance between two vectors
float KnnSearch::calculate_distance(const std::vector<float> &vec1, const std::vector<float> &vec2) const {
  if (vec1.size() != vec2.size()) {
    throw std::invalid_argument("Vector dimensions do not match");
  }

  float sum = 0.0;
  for (size_t i = 0; i < vec1.size(); ++i) {
    sum += std::pow(vec1[i] - vec2[i], 2);
  }
  return std::sqrt(sum);
}
