/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/10
 * Description: [Provide description here]
 */
#ifndef CANDY_INCLUDE_ALGORITHMS_SEARCH_ALGORITHM_HPP_
#define CANDY_INCLUDE_ALGORITHMS_SEARCH_ALGORITHM_HPP_
#include <vector>
#include <cstddef>  // For size_t

class SearchAlgorithm {
 public:
  // Virtual destructor
  virtual ~SearchAlgorithm() = default;

  // Insert a vector into the search algorithm's index
  virtual void insert(size_t id, const std::vector<float> &vec) = 0;

  // Query the nearest neighbors (returns vector of IDs)
  virtual std::vector<size_t> query(const std::vector<float> &query_vec, size_t k) const = 0;

  // Remove a vector from the search algorithm's index
  virtual void remove(size_t id) = 0;
};
#endif //CANDY_INCLUDE_ALGORITHMS_SEARCH_ALGORITHM_HPP_
