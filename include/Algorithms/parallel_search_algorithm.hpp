/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/11
 * Description: [Provide description here]
 */
#ifndef CANDY_INCLUDE_ALGORITHMS_PARALLEL_SEARCH_ALGORITHM_HPP_
#define CANDY_INCLUDE_ALGORITHMS_PARALLEL_SEARCH_ALGORITHM_HPP_

#include <Algorithms/search_algorithm.hpp>
#include <Parallelism//task_scheduler.hpp>
#include <vector>
#include <memory>
#include <vector>
#include <memory>


class ParallelSearchAlgorithm : public SearchAlgorithm {
 public:
  // Constructor
  ParallelSearchAlgorithm(TaskScheduler* scheduler, int ef_search = 10)
      : scheduler_(scheduler), ef_search_(ef_search) {}

  ~ParallelSearchAlgorithm() override = default;

  // Insert a vector into the index
  void insert(size_t id, const std::vector<float> &vec) override = 0;

  // Query the nearest neighbors (returns vector of IDs)
  std::vector<size_t> query(const std::vector<float> &query_vec, size_t k) const override = 0;

  // Remove a vector from the index
  void remove(size_t id) override = 0;

  void update(size_t id, const std::vector<float> &vector) override = 0;

protected:

  // Pure virtual method for searching neighbors (to be implemented by derived classes)
  virtual void search_layer(const std::vector<float>& query_vec, size_t k) const = 0;

  int ef_search_;
  TaskScheduler* scheduler_;  // Scheduler to manage parallel tasks
};


#endif //CANDY_INCLUDE_ALGORITHMS_PARALLEL_SEARCH_ALGORITHM_HPP_
