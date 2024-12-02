//
// Created by LIUJUN on 23/11/2024.
//
#include <torch/torch.h>
#include <vector>
#include "Algorithms/Utils/metric_type.hpp"
#ifndef VAMANA_BASE_HPP
#define VAMANA_BASE_HPP

class vertex {
 public:
  ~vertex() = default;
  idx_t id_;
  torch::Tensor vector_;
  std::vector<std::shared_ptr<vertex>> neighbors_;

  vertex(idx_t id, torch::Tensor vector) : id_(id), vector_(vector) {}
};

using VertexPtr = std::shared_ptr<vertex>;
#endif  //VAMANA_BASE_HPP