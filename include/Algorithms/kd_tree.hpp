//
// Created by zyt on 24-10-14.
//

#ifndef INTELLISTREAM_SRC_ALGORITHMS_KD_TREE_HPP_
#define INTELLISTREAM_SRC_ALGORITHMS_KD_TREE_HPP_

//#include <faiss/utils/distances.h>
//#include <flann/flann.hpp>
#include <torch/torch.h>

#include <algorithm>
#include <random>
#include <vector>

#include "search_algorithm.hpp"

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
class ResultSet {
  int64_t* indices;
  float* distances;
  int64_t size_;
  int64_t max_size;

 public:
  ResultSet(int64_t max_size) : max_size(max_size), size_(0) {
    indices = new int64_t[max_size];
    distances = new float[max_size];
    for (int i = 0; i < max_size; ++i) {
      distances[i] = std::numeric_limits<float>::max();
    }
  }
  ~ResultSet() {
    delete[] indices;
    delete[] distances;
  }

  size_t size() const { return this->size_; }
  float worstDist() const {
    if (size_ == 0) {
      return 0.0;
    }
    int64_t farthest = 0;
    for (int64_t i = 1; i < size_; i++) {
      if (distances[i] > distances[farthest]) {
        farthest = i;
      }
    }
    return distances[farthest];
  }

  bool isFull() const { return size_ == max_size; }
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

  void copy(int64_t* idx, float* dists, int64_t query_index, int64_t n) const {
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
  const BranchSt& top() const { return pq.top(); }
  bool empty() const { return pq.empty(); }
  size_t size() const { return pq.size(); }

 private:
  std::priority_queue<BranchSt, std::vector<BranchSt>, std::greater<BranchSt>>
      pq;
};

class VisitBitset {
 public:
  VisitBitset(size_t size) : bitset((size + 63) / 64, 0), size(size) {}
  void set(size_t index) { bitset[index / 64] |= (1ULL << (index % 64)); }

  bool test(size_t index) const {
    return (bitset[index / 64] & (1ULL << (index % 64))) != 0;
  }

  void clear() { std::fill(bitset.begin(), bitset.end(), 0); }

  size_t getSize() const { return size; }

 private:
  std::vector<uint64_t> bitset;
  size_t size;
};

class KDTree : public SearchAlgorithm {
 public:
  typedef Node* NodePtr;
  typedef BranchStruct<NodePtr> BranchSt;
  typedef BranchSt* Branch;

 private:
  /// Number of randomized trees that are used in forest
  uint64_t num_trees;
  float* mean;
  float* var;
  uint64_t ntotal;
  size_t vecDim;

  int RAND_DIM = 5;
  int SAMPLE_MEAN = 114;

  torch::Tensor dbTensor;
  int64_t lastNNZ;
  int64_t expandStep;
  float eps;
  int checks;
  std::vector<NodePtr> tree_roots;

 public:
  KDTree(size_t dimensions);

  // Insert a vector into the k-NN index
  void insert(size_t id, const std::vector<float>& vec);

  // Query k nearest neighbors (returns vector of IDs)
  std::vector<size_t> query(const std::vector<float>& query_vec, size_t k);

  // Remove a vector from the k-NN index
  void remove(size_t id);

  float fvec_L2sqr(const float* vec1, const float* vec2, int d);

  /**
   * @brief add dbTensor[idx] to tree with root as node
   * @param node typically a tree root
   * @param idx index in dbTensor
   */
  void addPointToTree(NodePtr node, int64_t idx);

  /**
   * @brief add data into the tree either by reconstruction or appending
   * @param t new data
   */
  void addPoints(torch::Tensor& t);

  /**
   * @brief perform knn-search on the kdTree structure
   * @param q query data to be searched
   * @param idx result vectors indices
   * @param distances result vectors' distances with query
   * @param aknn number of approximate neighbors
   * @return number of results obtained
   */
  int knnSearch(torch::Tensor& q, int64_t* idx, float* distances, int64_t aknn);

  /**
   * @brief
   * @param result
   * @param vec
   * @param maxCheck
   * @param epsError
   */
  void getNeighbors(ResultSet& result, const float* vec, int maxCheck,
                    float epsError);

  /**
   * @brief search from a given node of the tree
   * @param result priority queue to store results
   * @param vec vector to be searched
   * @param node current node to be traversed
   * @param mindist current minimum distance obtained
   * @param checkCount count of checks on multiple trees
   * @param maxCheck max check on multiple trees
   * @param epsError error to be compared with worst distance
   * @param heap heap structure to store branches
   * @param checked visited bitmap
   */
  void searchLevel(ResultSet& result, const float* vec, NodePtr node,
                   float mindist, int& checkCount, int maxCheck, float epsError,
                   Heap<BranchSt>* heap, VisitBitset& checked);

  /**
   * @brief build the tree from scratch
   */
  void buildTree();

  /**
   * @brief create a node that subdivides vectors from data[first] to data[last]. Called recursively on each subset
   * @param idx index of this vector
   * @param count number of vectors in this sublist
   * @return
   */
  NodePtr divideTree(int64_t* idx, int count);

  /**
   * @brief choose which feature to use to subdivide this subset of vectors by randomly choosing those with highest variance
   * @param ind index of this vector
   * @param count number of vectors in this sublist
   * @param index index where the sublist split
   * @param cutfeat index of highest variance as cut feature
   * @param cutval value of highest variance
   */
  void meanSplit(int64_t* ind, int count, int64_t& index, int64_t& cutfeat,
                 float& cutval);

  /**
   * @brief select top RAND_DIM largest values from vector and return index of one of them at random
   * @param v values of variance
   * @return the index of randomly chosen highest variance
   */
  int selectDivision(float* v);

  /**
   * @brief subdivide the lists by a plane perpendicular on axe corresponding to the cutfeat dimension at cutval position
   * @param ind index of the list
   * @param count count of the list
   * @param cutfeat the chosen feature
   * @param cutval the threshold value to be compared
   * @param lim1 split index candidate for meansplit
   * @param lim2 split index candidate for meansplit
   */
  void planeSplit(int64_t* ind, int count, int64_t cutfeat, float cutval,
                  int& lim1, int& lim2);
};
#endif  //INTELLISTREAM_SRC_ALGORITHMS_KD_TREE_HPP_
