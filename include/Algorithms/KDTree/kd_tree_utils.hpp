/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/25 21:53
 * Modified by:
 * Modified on:
 * Description: ${DESCRIPTION}
 */
#pragma once
#include <torch/torch.h>
#include <vector>
#include <algorithm>
#include <random>
#include <limits>
#include <queue>

#ifndef KD_TREE_UTILS_HPP
#define KD_TREE_UTILS_HPP
struct Node {
public:
    /// index used for subdivision.
    int64_t divfeat;
    /// The value used for subdivision
    float divval;
    /// Node data
    torch::Tensor data;
    Node *child1, *child2;

    Node() {
        child1 = nullptr;
        child2 = nullptr;
    }

    ~Node() {
        if (child1 != nullptr) {
            // child1->~Node();

            //delete could be a better way perhaps
            delete child1;
            child1 = nullptr;
        }
        if (child2 != nullptr) {
            // child2->~Node();
            delete child2;
            child2 = nullptr;
        }
    }
};
class ResultSet{
  int64_t* indices;
  float* distances;
  int64_t size_;
  int64_t max_size;

  public:
  ResultSet(int64_t max_size):max_size(max_size), size_(0){
    indices = new int64_t[max_size];
    distances = new float[max_size];
    for (int i = 0; i < max_size; ++i) {
      distances[i] = std::numeric_limits<float>::max();
    }
  }
  ~ResultSet(){
    delete[] indices;
    delete[] distances;
  }

  size_t size() const{
    return this->size_;
  }
  float worstDist() const{
    if(size_ == 0){
      return 0.0;
    }
    int64_t farthest = 0;
    for(int64_t i = 1; i < size_; i++){
      if (distances[i] > distances[farthest]) {
          farthest = i;
      }
    }
    return distances[farthest];
  }

  bool isFull() const {
    return size_ == max_size;
  }
  void add(float distance, int64_t index) {
    if (size_ < max_size) {
      indices[size_] = index;
      distances[size_] = distance;
      size_++;
    } else {
        int64_t farthest = 0;
        for (int64_t i = 0; i < max_size; i++) {
          if (distances[i] > distances[farthest]) {
            farthest = i;
          }
        }
      if (distance < distances[farthest]) {
        indices[farthest] = index;
        distances[farthest] = distance;
      }
      }
    }

  void copy(int64_t *idx, float *dists, int64_t query_index, int64_t n) const {
        for (int64_t i = 0; i < n; i++) {
            idx[query_index * max_size + i] = indices[i];
            dists[query_index * max_size + i] = distances[i];
        }
    }
};

template <typename NodePtr>
struct BranchStruct {
    NodePtr node;
    float mindist;

    // 构造函数
    BranchStruct(const NodePtr node_ = nullptr, float dist = 0.0f)
        : node(node_), mindist(dist) {}

    bool operator<(const BranchStruct& rhs) const {
       // reverse '<' to '>' as we need min heap when using std::priority_queue
        return mindist > rhs.mindist;
    }
    bool operator>(const BranchStruct& rhs) const {
        // reverse '>' to '<' as we need min heap when using std::priority_queue
        return mindist < rhs.mindist;
    }

};
template <typename BranchSt>
class Heap {
public:
    bool push(const BranchSt& branch) {
        pq.push(branch);
        return true;
    }
    bool pop(BranchSt& branch) {
        if (pq.empty()) {
            //throw std::out_of_range("Heap is empty");
            return false;
        }
        branch = pq.top();
        pq.pop();
        return true;
    }
    const BranchSt& top() const {
        return pq.top();
    }
    bool empty() const {
        return pq.empty();
    }
    size_t size() const {
        return pq.size();
    }

private:
    std::priority_queue<BranchSt, std::vector<BranchSt>,std::greater<BranchSt>> pq;
};

class VisitBitset {
public:
    VisitBitset(size_t size) : bitset((size + 63) / 64, 0), size(size) {}
    void set(size_t index) {
        bitset[index / 64] |= (1ULL << (index % 64));
    }

    bool test(size_t index) const {
        return (bitset[index / 64] & (1ULL << (index % 64))) != 0;
    }

    void clear() {
        std::fill(bitset.begin(), bitset.end(), 0);
    }

    size_t getSize() const {
        return size;
    }

private:
    std::vector<uint64_t> bitset;
    size_t size;
};

#endif //KD_TREE_UTILS_HPP
