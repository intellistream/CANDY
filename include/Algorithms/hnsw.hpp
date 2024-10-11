/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/9
 * Description: [Provide description here]
 */
#ifndef CANDY_INCLUDE_ALGORITHMS_HNSW_ALGORITHM_HPP_
#define CANDY_INCLUDE_ALGORITHMS_HNSW_ALGORITHM_HPP_

#include "search_algorithm.hpp"
#include <unordered_map>
#include <queue>
#include <cmath>
#include <limits>
#include <algorithm>
#include <memory>

class HNSWAlgorithm : public SearchAlgorithm {
 public:
  // Constructor
  HNSWAlgorithm(int max_level = 1, int ef_search = 10)
      : max_level_(max_level), ef_search_(ef_search), entry_point_(nullptr) {}

  // Destructor
  ~HNSWAlgorithm() override = default;

  // Insert a vector into the HNSW index
  void insert(size_t id, const std::vector<float> &vec) override {
    auto vertex = std::make_shared<Vertex>(id, vec);
    if (!entry_point_) {
      entry_point_ = vertex;
      max_level_ = 0;
    } else {
      int level = random_level();
      if (level > max_level_) {
        max_level_ = level;
        entry_point_ = vertex;
      }
      insert_at_level(vertex, level);
    }
    index_[id] = vertex;
  }

  // Query the nearest neighbors (returns vector of IDs)
  std::vector<size_t> query(const std::vector<float> &query_vec, size_t k) const override {
    if (!entry_point_) {
      return {};
    }

    std::priority_queue<std::pair<float, VertexPtr>> candidates;
    std::priority_queue<std::pair<float, VertexPtr>> top_candidates;

    auto nearest = entry_point_;
    float d_nearest = euclidean_distance(query_vec, nearest->vector);
    candidates.emplace(d_nearest, nearest);
    top_candidates.emplace(d_nearest, nearest);

    for (int level = max_level_; level > 0; --level) {
      nearest = greedy_update_nearest(query_vec, level, nearest, d_nearest);
    }

    search_layer(query_vec, k, candidates, top_candidates);

    // Collect the result IDs from the top candidates
    std::vector<size_t> result;
    while (!top_candidates.empty()) {
      result.push_back(top_candidates.top().second->id);
      top_candidates.pop();
    }
    std::reverse(result.begin(), result.end());
    return result;
  }

  // Remove a vector from the HNSW index
  void remove(size_t id) override {
    index_.erase(id);
  }

 private:
  struct Vertex {
    size_t id;
    std::vector<float> vector;
    std::vector<std::vector<std::shared_ptr<Vertex>>> neighbors;

    Vertex(size_t id, const std::vector<float> &vec)
        : id(id), vector(vec), neighbors() {}
  };

  using VertexPtr = std::shared_ptr<Vertex>;

  // Utility function to calculate Euclidean distance between two vectors
  float euclidean_distance(const std::vector<float> &a, const std::vector<float> &b) const {
    float sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
      float diff = a[i] - b[i];
      sum += diff * diff;
    }
    return std::sqrt(sum);
  }

  // Random level generator for new nodes
  int random_level() const {
    int level = 0;
    while (level < max_level_ && (rand() % 2) == 0) {
      ++level;
    }
    return level;
  }

  // Greedy search to find the nearest neighbor at a given level
  VertexPtr greedy_update_nearest(const std::vector<float> &query_vec, int level, VertexPtr nearest, float &d_nearest) const {
    bool changed;
    do {
      changed = false;
      for (const auto &neighbor : nearest->neighbors[level]) {
        float dist = euclidean_distance(query_vec, neighbor->vector);
        if (dist < d_nearest) {
          d_nearest = dist;
          nearest = neighbor;
          changed = true;
        }
      }
    } while (changed);
    return nearest;
  }

  // Search for neighbors at a specific layer
  void search_layer(const std::vector<float> &query_vec, size_t k, std::priority_queue<std::pair<float, VertexPtr>> &candidates, std::priority_queue<std::pair<float, VertexPtr>> &top_candidates) const {
    std::unordered_map<size_t, bool> visited;

    while (!candidates.empty()) {
      auto[d_curr, curr_vertex] = candidates.top();
      candidates.pop();

      for (const auto &neighbor : curr_vertex->neighbors[0]) {
        if (visited[neighbor->id]) {
          continue;
        }
        visited[neighbor->id] = true;

        float dist = euclidean_distance(query_vec, neighbor->vector);
        if (top_candidates.size() < k) {
          top_candidates.emplace(dist, neighbor);
          candidates.emplace(dist, neighbor);
        } else if (dist < top_candidates.top().first) {
          top_candidates.pop();
          top_candidates.emplace(dist, neighbor);
          candidates.emplace(dist, neighbor);
        }
      }
    }
  }

  // Insert a vertex at a given level
  void insert_at_level(VertexPtr vertex, int level) {
    auto current = entry_point_;
    float d_current = euclidean_distance(vertex->vector, current->vector);

    for (int lvl = max_level_; lvl > level; --lvl) {
      current = greedy_update_nearest(vertex->vector, lvl, current, d_current);
    }

    for (int lvl = level; lvl >= 0; --lvl) {
      std::vector<VertexPtr> &neighbors = current->neighbors[lvl];
      neighbors.push_back(vertex);
      vertex->neighbors.resize(level + 1);
      vertex->neighbors[lvl].push_back(current);
    }
  }

  // Index to store the vectors with their IDs
  std::unordered_map<size_t, VertexPtr> index_;
  VertexPtr entry_point_;
  int max_level_;
  int ef_search_;
};

#endif //CANDY_INCLUDE_ALGORITHMS_HNSW_ALGORITHM_HPP_
