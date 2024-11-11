/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Junyao Dong
 * Created on: 2024/10/24
 * Description:
 */

#ifndef CANDY_INCLUDE_ALGORITHMS_PARALLEL_HNSW_BASE_HPP_
#define CANDY_INCLUDE_ALGORITHMS_PARALLEL_HNSW_BASE_HPP_

#include "Algorithms/Utils/metric_type.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

template <typename T>
class SafeVector {
 public:
  explicit SafeVector(size_t n_size = 0) {
    if (n_size > 0) {
      m_vec.reserve(n_size);
    }
  }

  ~SafeVector() {
    std::lock_guard lock(m_mtx);
    m_vec.clear();
  }

  SafeVector& operator=(SafeVector const& other) {
    if (this != &other) {
      std::lock(m_mtx, other.m_mtx);
      std::lock_guard lock1(m_mtx, std::adopt_lock);
      std::lock_guard<std::mutex> lock2(other.m_mtx, std::adopt_lock);
      m_vec = other.m_vec;
    }
    return *this;
  }

  SafeVector& operator=(SafeVector&& other) noexcept {
    if (this != &other) {
      std::lock(m_mtx, other.m_mtx);
      std::lock_guard lock1(m_mtx, std::adopt_lock);
      std::lock_guard<std::mutex> lock2(other.m_mtx, std::adopt_lock);
      m_vec = std::move(other.m_vec);
    }
    return *this;
  }

  bool empty() const {
    std::lock_guard lock(m_mtx);
    return m_vec.empty();
  }

  std::size_t size() const {
    std::lock_guard lock(m_mtx);
    return m_vec.size();
  }

  std::size_t capacity() const {
    std::lock_guard lock(m_mtx);
    return m_vec.capacity();
  }

  T& at(std::size_t index) {
    std::lock_guard lock(m_mtx);
    return m_vec.at(index);
  }

  const T& at(std::size_t index) const {
    std::lock_guard lock(m_mtx);
    return m_vec.at(index);
  }

  void push_back(const T& element) {
    std::unique_lock lock(m_mtx);
    m_vec.push_back(element);
    lock.unlock();
  }

  void pop_back() {
    std::unique_lock lock(m_mtx);
    m_vec.pop_back();
    lock.unlock();
  }

  template <typename... Args>
  void emplace_back(Args&&... args) {
    std::unique_lock lock(m_mtx);
    m_vec.emplace_back(std::forward<Args>(args)...);
  }

  void reserve(std::size_t new_capacity) {
    std::unique_lock lock(m_mtx);
    m_vec.reserve(new_capacity);
  }

  void clear() {
    std::unique_lock lock(m_mtx);
    m_vec.clear();
  }

  T& front() {
    std::lock_guard lock(m_mtx);
    if (m_vec.empty()) {
      throw std::out_of_range("Vector is empty");
    }
    return m_vec.front();
  }

  T& back() {
    std::lock_guard lock(m_mtx);
    if (m_vec.empty()) {
      throw std::out_of_range("Vector is empty");
    }
    return m_vec.back();
  }

  auto begin() {
    std::lock_guard lock(m_mtx);
    return m_vec.begin();
  }

  auto end() {
    std::lock_guard lock(m_mtx);
    return m_vec.end();
  }

  T& operator[](std::size_t index) {
    std::lock_guard lock(m_mtx);
    return m_vec[index];
  }

 private:
  std::vector<T> m_vec;

  mutable std::mutex m_mtx;
};

class Vertex;

struct Neighbors {
  Vertex* vertex;
  float distance;
  int vertex_label;

  Neighbors(const Neighbors& n) {
    this->distance = n.distance;
    this->vertex = n.vertex;
    this->vertex_label = n.vertex_label;
  }

  Neighbors(Vertex* n, const float distance, const int unique_id) {
    this->vertex = n;
    this->distance = distance;
    this->vertex_label = unique_id;
  }
};

class Vertex {
 public:
  std::vector<std::vector<idx_t>> neighbors_{};

  Vertex(long level) : neighbors_(level + 1) {}

  Vertex() = default;

  ~Vertex() = default;
};

class VisitedList {
 public:
  VisitedList() : cur_version_(0) {}

  void init(const size_t num_elements) {
    cur_version_ = -1;
    visited_flags_.assign(num_elements, 0);
  }

  void reset() {
    cur_version_++;
    if (cur_version_ == 0) {
      std::ranges::fill(visited_flags_, 0);
      cur_version_++;
    }
  }

  void mark_visited(const int index) { visited_flags_[index] = cur_version_; }

  void mark_visited(const long index) { visited_flags_[index] = cur_version_; }

  [[nodiscard]] bool is_visited(const int index) const {
    return visited_flags_[index] == cur_version_;
  }

 private:
  int cur_version_;
  std::vector<int> visited_flags_;
};

struct NeighborCmpNearest {
  bool operator()(const Neighbors& a, const Neighbors& b) const {
    return a.distance < b.distance;
  }
};

struct NeighborCmpFarthestHeap {
  bool operator()(const Neighbors& a, const Neighbors& b) const {
    return a.distance < b.distance;
  }
};

struct NeighborCmpNearestHeap {
  bool operator()(const Neighbors& a, const Neighbors& b) const {
    return a.distance > b.distance;
  }
};

struct CompareByDistInTupleHeap {
  constexpr bool operator()(
      const std::tuple<Vertex*, size_t>& i1,
      const std::tuple<Vertex*, size_t>& i2) const noexcept {
    return std::get<1>(i1) > std::get<1>(i2);
  }
};

#endif  //CANDY_INCLUDE_ALGORITHMS_PARALLEL_HNSW_BASE_HPP_