/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/9
 * Description: [Provide description here]
 */
#ifndef INTELLISTREAM_SRC_ALGORITHMS_KNN_SEARCH_HPP_
#define INTELLISTREAM_SRC_ALGORITHMS_KNN_SEARCH_HPP_

#include "search_algorithm.hpp"
#include <unordered_map>
#include <vector>
#include <cmath>  // For distance calculations

class KnnSearch : public SearchAlgorithm {
 public:
 // Destructor
 ~KnnSearch() override = default;

 // Constructor with vector dimensions
 KnnSearch(size_t dimensions);

 // Insert a vector into the k-NN index
 void insert(size_t id, const std::vector<float> &vec) override;

 // Query k nearest neighbors (returns vector of IDs)
 std::vector<size_t> query(const std::vector<float> &query_vec, size_t k) const override;

 // Remove a vector from the k-NN index
 void remove(size_t id) override;

 void update(size_t id, const std::vector<float> &vector) override;

private:
 size_t dimensions;
 std::unordered_map<size_t, std::vector<float> > index;

 // Helper function to calculate Euclidean distance
  float calculate_distance(const std::vector<float> &vec1, const std::vector<float> &vec2) const;
};

#endif //INTELLISTREAM_SRC_ALGORITHMS_KNN_SEARCH_HPP_
