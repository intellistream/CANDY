/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Junyao Dong
 * Created on: 2024/10/22
 * Description: Parallel-HNSW version 1: using coarse-grained locking insert
 */

#ifndef CANDY_INCLUDE_ALGORITHMS_PARALLEL_HNSW_V1_HPP_
#define CANDY_INCLUDE_ALGORITHMS_PARALLEL_HNSW_V1_HPP_

#include <torch/torch.h>
#include <Utils/ConfigMap.hpp>
#include <random>
#include <unordered_set>
#include <vector>
#include <IO/BasicStorage.hpp>

#include "Algorithms/ANNSBase.hpp"
#include "Algorithms/HNSW/HNSWBase.hpp"

namespace CANDY_ALGO {
class HNSW final : public ANNSBase {
 public:
  HNSW() = default;

  ~HNSW() override = default;

  bool setConfig(INTELLI::ConfigMapPtr cfg) override;

  void reset() override;

  struct distAndId {
    float dist;
    idx_t id;

    distAndId(const float d, const idx_t i) : dist(d), id(i) {}

    bool operator<(const distAndId& other) const { return dist < other.dist; }

    bool operator>(const distAndId& other) const { return dist > other.dist; }
  };

  struct compByDistLess {
    bool operator()(const distAndId& a, const distAndId& b) const {
      return a.dist < b.dist;
    }
  };

  struct compByDistGreater {
    bool operator()(const distAndId& a, const distAndId& b) const {
      return a.dist > b.dist;
    }
  };

  using priority_of_distAndId_Less =
      std::priority_queue<distAndId, std::vector<distAndId>, compByDistLess>;

  // Search layer on non-zero level
  priority_of_distAndId_Less search_base_layer(idx_t, const torch::Tensor&,
                                               long);

  priority_of_distAndId_Less search_base_layerST(idx_t, const torch::Tensor&,
                                                 long);

  int random_level();

  void get_neighbors_by_heuristic(priority_of_distAndId_Less& top_candidates,
                                  int64_t M);

  void create_link(idx_t from, idx_t to, long level, bool link_double = true);

  void remove_link(idx_t from, idx_t to, long level, bool link_double = false);

  long mutually_connect_new_element(const torch::Tensor& tensor, int64_t id,
                                    priority_of_distAndId_Less& top_candidates,
                                    long level, bool is_update);

  idx_t fetch_free_idx();

  void insert(const int vid);

  bool insertTensor(const torch::Tensor& t) override;

  void remove(idx_t idx);

  auto deleteTensor(torch::Tensor& t, int64_t k) -> bool override;

  bool reviseTensor(torch::Tensor& t, torch::Tensor& w) override;

  torch::Tensor search(const torch::Tensor& tensor, int64_t k);

  std::vector<torch::Tensor> searchTensor(const torch::Tensor& q,
                                          int64_t k) override;
                                  
  bool loadInitialTensor(torch::Tensor& t) override;

  inline const torch::Tensor get_vector_by_vid(int id) {
    return storage_engine.getVectorByVid(db_vids_[id]);
  }

 protected:
  int64_t M_{};
  int64_t Mmax_{};
  int64_t Mmax0_{};
  int64_t efConstruction_{};
  int64_t efSearch_{};

  long size_ = -1;  // how many vectors in the db

  long initialVolume_{};  
  long dim_{};
  static constexpr int expandStep = 1000;

  vector<int> db_vids_;

  long entry_point_ = -1;
  long max_level_ = 0;

  std::vector<Vertex> vertexes_;
  std::vector<idx_t> free_list_;
  std::default_random_engine level_generator_;

  BasicStorage storage_engine;

  std::mutex mtx;
};
}  // namespace CANDY_ALGO
#endif