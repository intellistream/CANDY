/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Junyao Dong
 * Created on: 2024/10/22
 * Description:
 */

#include <Algorithms/HNSW/hnsw.hpp>
#include <Utils/Computation.hpp>
#include <IO/BasicStorage.hpp>
#include <Utils/TensorOP.hpp>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <memory>
#include <unordered_set>
#include <vector>
#include "Algorithms/Utils/metric_type.hpp"
using namespace CANDY_ALGO;

bool HNSW::setConfig(const INTELLI::ConfigMapPtr cfg) {
  if (cfg == nullptr) {
    return false;
  }
  // set hnsw parameters
  M_ = cfg->tryI64("M", 8, true);
  Mmax_ = cfg->tryI64("Mmax", 16, true);
  efConstruction_ = cfg->tryI64("efConstruction", 50, true);
  efSearch_ = cfg->tryI64("efSearch", 200, true);
  Mmax0_ = Mmax_ * 2;
  // set database parameters
  initialVolume_ = cfg->tryI64("initialVolume", 1000, true);
  dim_ = cfg->tryI64("vecDim", 768, true);
  dbTensor_ = torch::zeros({initialVolume_, dim_});
  size_ = -1;
  return true;
}

void HNSW::reset() {}

using distAndId = HNSW::distAndId;
using compByDistLess = HNSW::compByDistLess;
using compByDistGreater = HNSW::compByDistGreater;
using priority_of_distAndId_Less =
    std::priority_queue<distAndId, std::vector<distAndId>, compByDistLess>;

priority_of_distAndId_Less HNSW::search_base_layer(idx_t ep_id,
                                                   const torch::Tensor& tensor,
                                                   const long layer) {
  priority_of_distAndId_Less top_candidates;
  priority_of_distAndId_Less candidates;
  float lowerBound = CANDY::euclidean_distance(dbTensor_[ep_id].contiguous(),
                                               tensor.contiguous());
  top_candidates.emplace(lowerBound, ep_id);
  candidates.emplace(-lowerBound, ep_id);
  std::unordered_set<idx_t> visited;
  visited.insert(ep_id);
  while (!candidates.empty()) {
    const auto c = candidates.top();
    if (-c.dist > lowerBound && top_candidates.size() == efConstruction_) {
      break;
    }
    candidates.pop();
    for (const auto& neighbor : vertexes_[c.id].neighbors_[layer]) {
      if (visited.contains(neighbor))
        continue;
      visited.insert(neighbor);
      if (float dist = CANDY::euclidean_distance(
              dbTensor_[neighbor].contiguous(), tensor.contiguous());
          top_candidates.size() < efConstruction_ || dist < lowerBound) {
        candidates.emplace(-dist, neighbor);
        top_candidates.emplace(dist, neighbor);
        if (top_candidates.size() > efConstruction_) {
          top_candidates.pop();
        }
        if (!top_candidates.empty()) {
          lowerBound = top_candidates.top().dist;
        }
      }
    }
  }
  return top_candidates;
}

priority_of_distAndId_Less HNSW::search_base_layerST(idx_t ep,
                                                     const torch::Tensor& q,
                                                     const long ef) {
  priority_of_distAndId_Less top_candidates;
  priority_of_distAndId_Less candidates;
  float lowerBound = CANDY::euclidean_distance(dbTensor_[ep], q);
  candidates.emplace(-lowerBound, ep);
  top_candidates.emplace(lowerBound, ep);
  std::unordered_set<idx_t> visited;
  visited.insert(ep);
  while (!candidates.empty()) {
    auto [d, i] = candidates.top();
    if (const auto dist2query = -d;
        dist2query > lowerBound && top_candidates.size() == ef) {
      break;
    }
    candidates.pop();
    for (const auto& neighbor : vertexes_[i].neighbors_[0]) {
      if (visited.contains(neighbor))
        continue;
      visited.insert(neighbor);
      if (const auto dist = CANDY::euclidean_distance(
              dbTensor_[neighbor].contiguous(), q.contiguous());
          top_candidates.size() < ef || dist < lowerBound) {
        candidates.emplace(-dist, neighbor);
        top_candidates.emplace(dist, neighbor);
        if (top_candidates.size() > ef) {
          top_candidates.pop();
        }
        if (!top_candidates.empty()) {
          lowerBound = top_candidates.top().dist;
        }
      }
    }
  }
  return top_candidates;
}

int HNSW::random_level() {
  static double m = 1 / log(1.0 * static_cast<double>(M_));
  static std::uniform_real_distribution distribution(0.0, 1.0);
  const double r = -log(distribution(level_generator_)) * m;
  return static_cast<int>(r);
}

void HNSW::getNeighborsByHeuristic2(priority_of_distAndId_Less& top_candidates,
                                    const int64_t M) const {
  if (top_candidates.size() < M) {
    return;
  }
  priority_of_distAndId_Less candidates;
  std::vector<distAndId> needed;
  needed.reserve(M);
  auto db_tensor = dbTensor_.contiguous();
  while (!top_candidates.empty()) {
    auto [dist, idx] = top_candidates.top();
    candidates.emplace(-dist, idx);
    top_candidates.pop();
  }
  while (!candidates.empty()) {
    if (needed.size() >= M) {
      break;
    }
    auto [dist, idx] = candidates.top();
    const float dist2query = -dist;
    candidates.pop();
    bool good = true;
    for (auto& [d, i] : needed) {
      if (const float cur_dist = CANDY::euclidean_distance(
              db_tensor[i].contiguous(), db_tensor[idx].contiguous());
          cur_dist < dist2query) {
        good = false;
        break;
      }
    }
    if (good) {
      needed.emplace_back(dist, idx);
    }
  }
  for (const auto& [dist, idx] : needed) {
    top_candidates.emplace(-dist, idx);
  }
}

void HNSW::create_link(const idx_t from, const idx_t to, const long level,
                       const bool link_double) {
  auto M_cur_max = level ? Mmax_ : Mmax0_;
  if (auto& neighbors = vertexes_[from].neighbors_[level];
      neighbors.size() < M_cur_max) {
    neighbors.push_back(to);
  } else {
    throw std::runtime_error("neighbors.size() >= M_cur_max");
  }
  if (link_double) {
    if (auto& neighbors = vertexes_[to].neighbors_[level];
        neighbors.size() < M_cur_max) {
      neighbors.push_back(from);
    } else {
      throw std::runtime_error("neighbors.size() >= M_cur_max");
    }
  }
}

void HNSW::remove_link(const idx_t from, const idx_t to, const long level,
                       const bool link_double) {
  if (auto& neighbors = vertexes_[from].neighbors_[level]; !neighbors.empty()) {
    std::erase(neighbors, to);
  }
  if (link_double) {
    if (auto& neighbors = vertexes_[to].neighbors_[level]; !neighbors.empty()) {
      std::erase(neighbors, from);
    }
  }
}

long HNSW::mutually_connect_new_element(
    const torch::Tensor& tensor, int64_t id,
    priority_of_distAndId_Less& top_candidates, const long level,
    const bool is_update) {
  const auto M_cur_max = level ? Mmax_ : Mmax0_;
  getNeighborsByHeuristic2(top_candidates, M_);
  if (top_candidates.size() > M_)
    throw std::runtime_error("top_candidates.size() > M_");
  std::vector<idx_t> selected_neighbors;
  selected_neighbors.reserve(top_candidates.size());
  while (!top_candidates.empty()) {
    selected_neighbors.push_back(top_candidates.top().id);
    top_candidates.pop();
  }
  const idx_t next_nearest_ep = selected_neighbors.back();
  for (const auto& neighbor : selected_neighbors) {
    if (auto& neighbors = vertexes_[neighbor].neighbors_[level];
        neighbors.size() < M_cur_max) {
      create_link(id, neighbor, level, true);
    } else {
      assert(neighbors.size() == M_cur_max);
      /**
          * here is the reason why the vertex is not double linked
         */
      // find the farthest neighbor to replace
      float max_dist =
          CANDY::euclidean_distance(dbTensor_[id], dbTensor_[neighbor]);
      priority_of_distAndId_Less candidates;
      candidates.emplace(max_dist, id);
      std::unordered_set<idx_t> set{id};
      for (const auto& n_n : neighbors) {
        candidates.emplace(CANDY::euclidean_distance(dbTensor_[n_n], tensor),
                           n_n);
        set.insert(n_n);  // supposed neighbors
      }
      getNeighborsByHeuristic2(candidates, M_cur_max);
      assert(neighbors.size() >= candidates.size());
      int index = 0;
      while (!candidates.empty()) {
        set.erase(candidates.top().id);  // the rest of the unselected neighbors
        neighbors[index++] = candidates.top().id;
        candidates.pop();
      }
      if (auto it = set.find(id);
          it ==
          set.end()) {  // new element is the closer one, so it could insert into the neighbors
        create_link(id, neighbor, level, false);
      }
      // for (const auto& n : set) {
      //   remove_link(n, neighbor, level);
      // }
      // #TODD: if add acc will be lower
    }
  }
  return next_nearest_ep;
}

idx_t HNSW::fetch_free_idx() {
  if (free_list_.empty()) {
    return size_;
  }
  const auto idx = free_list_.back();
  free_list_.pop_back();
  return idx;
}

void HNSW::insert(const torch::Tensor& t) {
  auto insert_pos = fetch_free_idx();
  const long cur_level = random_level();
  auto db_tensor = dbTensor_.contiguous();
  auto tensor = t.contiguous();
  if (insert_pos >= size_) {
    // make a container to store the tensor
    torch::Tensor container = torch::zeros({1, dim_});
    container[0] = tensor;
    INTELLI::TensorOP::appendRowsBufferMode(
        &dbTensor_, &container, &size_,
        expandStep);  // append t to dbTensor_
    vertexes_.emplace_back(cur_level);
    insert_pos = size_;
  } else {
    db_tensor[insert_pos] = tensor;  // insert tensor to dbTensor_
    vertexes_[insert_pos].neighbors_.resize(cur_level + 1);
  }

  auto curr_obj = entry_point_;
  if (entry_point_ != -1) {
    if (cur_level < max_level_) {
      float cur_dist = CANDY::euclidean_distance(
          db_tensor[entry_point_].contiguous(), tensor.contiguous());
      std::unordered_set<idx_t> visited;
      for (int level = static_cast<int>(max_level_); level > cur_level;
           level--) {
        visited.clear();
        visited.insert(curr_obj);
        bool changed = true;
        while (changed) {
          changed = false;
          for (auto& neighbors = vertexes_[curr_obj].neighbors_[level];
               const auto& neighbor : neighbors) {
            if (visited.contains(neighbor))
              continue;
            visited.insert(neighbor);
            if (const float d = CANDY::euclidean_distance(
                    db_tensor[neighbor].contiguous(), tensor.contiguous());
                d < cur_dist) {
              cur_dist = d;
              curr_obj = neighbor;
              changed = true;
            }
          }
        }
      }
    }
    for (long level = std::min(cur_level, max_level_); level >= 0; level--) {
      auto top_candidates = search_base_layer(curr_obj, tensor, level);
      curr_obj = mutually_connect_new_element(tensor.contiguous(), insert_pos,
                                              top_candidates, level, false);
    }
  } else {
    entry_point_ = 0;
    max_level_ = cur_level;
  }
  if (cur_level > max_level_) {
    entry_point_ = insert_pos;
    max_level_ = cur_level;
  }
}

bool HNSW::insertTensor(const torch::Tensor& t) {
  mtx.lock();
  
  int 

  for (int64_t i = 0; i < t.size(0); i++) {
    insert(t[i]);
  }
  mtx.unlock();
  return true;
}

void HNSW::remove(const idx_t idx) {
  free_list_.push_back(idx);
  //TODO: if the vertex is the entry point, we need to update the entry point
  const bool is_entry_point = idx == entry_point_;
  idx_t new_entry_point = -1;
  auto& top_neighbors =
      vertexes_[idx].neighbors_[max_level_];  // size should not be 0
  while (is_entry_point && max_level_ > 0) {
    for (const auto& neighbor : top_neighbors) {
      if (new_entry_point == -1) {
        new_entry_point = neighbor;  // neighbor will bot be -1;
      } else {
        if (CANDY::euclidean_distance(dbTensor_[neighbor], dbTensor_[idx]) <
            CANDY::euclidean_distance(dbTensor_[new_entry_point],
                                      dbTensor_[idx])) {
          new_entry_point = neighbor;
        }
      }
    }
    if (new_entry_point != -1) {
      break;
    }
    top_neighbors = vertexes_[new_entry_point].neighbors_[--max_level_];
  }
  if (is_entry_point) {
    entry_point_ = new_entry_point;
  }
  auto& all_neighbors = vertexes_[idx].neighbors_;
  for (auto& neighbors_level : all_neighbors) {
    for (const auto& neighbor : neighbors_level) {
      for (auto& n_neighbors = vertexes_[neighbor].neighbors_;
           auto& n_neighbor : n_neighbors) {
        std::erase(n_neighbor, idx);
      }
    }
    neighbors_level.clear();
  }
  all_neighbors.clear();
}

bool HNSW::deleteTensor(torch::Tensor& t, const int64_t k) {
  // Use the searchTensor function to get the indices of k-nearest neighbors for each row in t
  const std::vector<torch::Tensor> idxToDeleteTensors = searchTensor(t, k);

  // Flatten idxToDeleteTensors into a single vector of int64_t
  std::vector<int64_t> idxToDelete;
  for (const auto& tensor : idxToDeleteTensors) {
    auto tensorAccessor = tensor.accessor<int64_t, 1>();
    for (int64_t i = 0; i < tensor.size(0); ++i) {
      idxToDelete.push_back(tensorAccessor[i]);
    }
  }
  // we use mark to delete the tensor
  for (const auto& idx : idxToDelete) {
    remove(idx);
  }

  return true;
}

bool HNSW::reviseTensor(torch::Tensor& t, torch::Tensor& w) {
  // not only revise the tensor but also the vertexes
  // Check if dimensions match
  if (t.size(0) > w.size(0) || t.size(1) != w.size(1)) {
    return false;
  }
  const auto idx2reviseTensors = searchTensor(t, 1);
  std::vector<int64_t> idx2revise;
  for (auto& tensor : idx2reviseTensors) {
    auto tensorAccessor = tensor.accessor<int64_t, 1>();
    for (int64_t i = 0; i < tensor.size(0); ++i) {
      auto idx = tensorAccessor[i];
      idx2revise.push_back(idx);
    }
  }
  int index = 0;
  for (const auto& idx : idx2revise) {
    remove(idx), insert(w[index++]);
  }
  return true;
}

torch::Tensor HNSW::search(const torch::Tensor& t, int64_t k) {
  // Store the indices of top-k nearest neighbors for this query row
  torch::Tensor result = torch::zeros({k}, torch::kInt64);
  auto tensor = t.contiguous();
  auto db_tensor = dbTensor_.contiguous();
  idx_t curr_obj = entry_point_;
  float curr_dist = CANDY::euclidean_distance(
      tensor.contiguous(), db_tensor[entry_point_].contiguous());
  for (long level = max_level_; level > 0; --level) {
    bool changed = true;
    while (changed) {
      changed = false;
      for (auto& neighbors = vertexes_[curr_obj].neighbors_[level];
           const auto neighbor : neighbors) {
        if (const float dist = CANDY::euclidean_distance(
                tensor.contiguous(), db_tensor[neighbor].contiguous());
            dist < curr_dist) {
          curr_dist = dist;
          curr_obj = neighbor;
          changed = true;
        }
      }
    }
  }
  auto top_candidates =
      search_base_layerST(curr_obj, tensor, std::max(efSearch_, k));
  while (top_candidates.size() > k) {
    top_candidates.pop();
  }
  long index = k - 1;
  while (!top_candidates.empty()) {
    auto [_, id] = top_candidates.top();
    result.index_put_({index--}, id);
    top_candidates.pop();
  }
  return result;
}

std::vector<torch::Tensor> HNSW::searchTensor(const torch::Tensor& q, 
                                              const int64_t k) {
  mtx.lock();
  const auto to_search = q.size(0);
  std::vector<torch::Tensor> to_return;
  to_return.reserve(to_search);
  if (size_ == 0)
    return {};
  for (int64_t i = 0; i < to_search; i++) {
    to_return.emplace_back(search(q[i], k));
  }
  mtx.unlock();
  return to_return;
}

bool HNSW::loadInitialTensor(torch::Tensor& t) {
  auto preSize = size_ + 1;
  INTELLI::TensorOP::appendRowsBufferMode(&dbTensor_, &t, &size_, expandStep);
  // append t to dbTensor_
  for (int64_t insert_pos = preSize; insert_pos <= size_; insert_pos++) {
    long cur_level = random_level();
    vertexes_.emplace_back(cur_level);
    auto tensor = dbTensor_[insert_pos].contiguous();
    auto db_tensor = dbTensor_.contiguous();
    auto curr_obj = entry_point_;
    if (entry_point_ != -1) {
      if (cur_level < max_level_) {
        float cur_dist = CANDY::euclidean_distance(
            db_tensor[entry_point_].contiguous(), tensor.contiguous());
        std::unordered_set<idx_t> visited;
        for (int level = static_cast<int>(max_level_); level > cur_level;
             level--) {
          visited.clear();
          visited.insert(curr_obj);
          bool changed = true;
          while (changed) {
            changed = false;
            for (auto neighbors = vertexes_[curr_obj].neighbors_[level];
                 const auto& neighbor : neighbors) {
              if (visited.contains(neighbor))
                continue;
              visited.insert(neighbor);
              if (const float d = CANDY::euclidean_distance(
                      db_tensor[neighbor].contiguous(), tensor.contiguous());
                  d < cur_dist) {
                cur_dist = d;
                curr_obj = neighbor;
                changed = true;
              }
            }
          }
        }
      }
      for (long level = std::min(cur_level, max_level_); level >= 0; level--) {
        auto top_candidates = search_base_layer(curr_obj, tensor, level);
        curr_obj = mutually_connect_new_element(tensor.contiguous(), insert_pos,
                                                top_candidates, level, false);
      }
    } else {
      entry_point_ = 0;
      max_level_ = cur_level;
    }
    if (cur_level > max_level_) {
      entry_point_ = insert_pos;
      max_level_ = cur_level;
    }
  }
  return true;
}