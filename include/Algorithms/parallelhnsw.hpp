/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/11
 * Description: [Provide description here]
 */
#ifndef CANDY_INCLUDE_ALGORITHMS_PARALLEL_HNSW_ALGORITHM_HPP_
#define CANDY_INCLUDE_ALGORITHMS_PARALLEL_HNSW_ALGORITHM_HPP_

#include <Algorithms/parallel_search_algorithm.hpp>
#include <vector>
#include <queue>
#include <memory>
#include <unordered_map>

class ParallelHNSWAlgorithm : public ParallelSearchAlgorithm {
 public:
  ParallelHNSWAlgorithm(TaskScheduler* scheduler, int max_level = 1, int ef_search = 10)
      : ParallelSearchAlgorithm(scheduler, ef_search), max_level_(max_level), entry_point_(nullptr) {}

  void insert(size_t id, const std::vector<float>& vec) override {
    // HNSW-specific insertion logic, using multiple levels
  }

  std::vector<size_t> query(const std::vector<float>& query_vec, size_t k) const override {
    // HNSW-specific query logic, using the graph to find nearest neighbors
    return {};
  }

  void remove(size_t id) override {
    // HNSW-specific removal logic
  }

 protected:
  void search_layer(const std::vector<float>& query_vec, size_t k) const override {
    // HNSW-specific search layer logic
  }

  struct Vertex {
    size_t id;
    std::vector<float> vector;
    std::vector<std::vector<std::shared_ptr<Vertex>>> neighbors;

    Vertex(size_t id, const std::vector<float>& vec)
        : id(id), vector(vec), neighbors() {}
  };

  using VertexPtr = std::shared_ptr<Vertex>;
  std::unordered_map<size_t, VertexPtr> index_;
  VertexPtr entry_point_;
  int max_level_;
};

#endif // CANDY_INCLUDE_ALGORITHMS_PARALLEL_HNSW_ALGORITHM_HPP_

