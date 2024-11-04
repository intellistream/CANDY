/*
* Copyright (C) 2024 by the INTELLI team
 * Created on: 2024/10/14
 * Description: [Provide description here]
 */
#include <Algorithms/KDTree/KDTree.hpp>
#include <Utils/Computation.hpp>
#include <torch/torch.h>
#include <vector>
#include <algorithm>
#include <random>
#include <Utils/TensorOP.hpp>

#include "Utils/Param.hpp"

namespace CANDY_ALGO {
bool KDTree::setConfig(INTELLI::ConfigMapPtr cfg) {
  vecDim = cfg->tryI64("vecDim", 768, true);
  num_trees = cfg->tryI64("numTrees", 4, true);
  tree_roots = std::vector<NodePtr>(num_trees, nullptr);
  ANNSBase::setConfig(cfg);
  return true;
}

bool KDTree::setParams(CANDY::ParamPtr param) {
  num_trees = param->num_trees;
  printf("Best param for KdTree\n num_trees: %ld\n", num_trees);
  return true;

}

KDTree::KDTree(size_t dimensions) : vecDim(dimensions), mean(nullptr), var(nullptr), lastNNZ(-1), expandStep(100),
                                    eps(0.0), checks(32), ntotal(0) {

  vecDim = 768; // default
  num_trees = 4; // default
  dbTensor = torch::zeros({0, static_cast<int64_t>(vecDim)});
  tree_roots = std::vector<KDTree::NodePtr>(num_trees, nullptr);
}

KDTree::~KDTree() {
  // Clean up tree roots
  for (NodePtr root : tree_roots) {
    delete root;
  }
  delete[] mean;
  delete[] var;
}

bool KDTree::insertTensor(const torch::Tensor &t) {
  addPoints(const_cast<torch::Tensor &>(t));
  return true;
}

bool KDTree::loadInitialTensor(const torch::Tensor &t) {
  dbTensor = t;
  buildTree();
  return true;
}

bool KDTree::deleteTensor(const torch::Tensor &t, int64_t k) {
  // Implementation for deleting tensor
  return true;
}

bool KDTree::reviseTensor(const torch::Tensor &t, const torch::Tensor &w) {
  // Implementation for revising tensor
  return true;
}

std::vector<idx_t> KDTree::searchIndex(const torch::Tensor &q, int64_t k) {
  int64_t query_size = q.size(0);
  std::vector<idx_t> results(query_size * k);
  float *distances = new float[query_size * k];
  knnSearch(const_cast<torch::Tensor &>(q), results.data(), distances, k);
  delete[] distances;
  return results;
}

torch::Tensor KDTree::rawData() {
  return dbTensor;
}

std::vector<torch::Tensor> KDTree::searchTensor(const torch::Tensor &q, int64_t k) {
  int64_t query_size = q.size(0);
  std::vector<torch::Tensor> results(query_size);
  int64_t *indices = new int64_t[query_size * k];
  float *distances = new float[query_size * k];
  knnSearch(const_cast<torch::Tensor &>(q), indices, distances, k);

  for (int64_t i = 0; i < query_size; ++i) {
    results[i] = torch::from_blob(indices + i * k, {k}, torch::kInt64).clone();
  }
  delete[] indices;
  delete[] distances;
  return results;
}

void KDTree::addPoints(torch::Tensor &t) {
  bool success = INTELLI::TensorOP::appendRowsBufferMode(&dbTensor, &t, &lastNNZ, expandStep);
  assert(success);
  ntotal += t.size(0);
  if ((ntotal - t.size(0)) * 2 < ntotal) {
    buildTree();
  } else {
    for (uint64_t i = ntotal - t.size(0); i < ntotal; i++) {
      for (uint64_t j = 0; j < num_trees; j++) {
        addPointToTree(tree_roots[j], i);
      }
    }
  }
}

int KDTree::knnSearch(torch::Tensor &q, int64_t *idx, float *distances, int64_t aknn) {
  int count = 0;
  for (int64_t i = 0; i < q.size(0); i++) {
    ResultSet resultSet = ResultSet(aknn);
    auto query_data = q.slice(0, i, i + 1).contiguous().data_ptr<float>();
    get_neighbors(resultSet, query_data, checks, eps + 1);
    int64_t n = std::min(resultSet.size(), static_cast<size_t>(aknn));
    resultSet.copy(idx, distances, i, n);
    count += n;
  }
  return count;
}

KDTree::NodePtr KDTree::divideTree(int64_t *ind, int count) {
  NodePtr node = new Node();

  if (count == 1) {
    node->child1 = nullptr;
    node->child2 = nullptr;
    node->divfeat = *ind;
    node->data = dbTensor[*ind];
  } else {
    int64_t idx;
    int64_t cutfeat;
    float cutval;
    meanSplit(ind, count, idx, cutfeat, cutval);
    node->divfeat = cutfeat;
    node->divval = cutval;
    //printf("starting  left dividing points with count=%d idx=%d\n", count, idx);
    node->child1 = divideTree(ind, idx);
    //printf("starting  right dividing points with count=%d idx=%d\n", count, idx);
    node->child2 = divideTree(ind + idx, count - idx);
  }

  return node;
}

void KDTree::meanSplit(int64_t *ind, int count, int64_t &index, int64_t &cutfeat, float &cutval) {
  memset(mean, 0, vecDim * sizeof(float));
  memset(var, 0, vecDim * sizeof(float));
  int cnt = std::min(KDTree::SAMPLE_MEAN, count);
  for (int j = 0; j < cnt; j++) {
    float *v = dbTensor.slice(0, ind[j], ind[j + 1]).contiguous().data_ptr<float>();
    for (int64_t k = 0; k < vecDim; k++) {
      mean[k] += v[k];
    }
  }
  float div_factor = 1.0 / cnt;
  for (int64_t k = 0; k < vecDim; k++) {
    mean[k] *= div_factor;
  }
  for (int j = 0; j < cnt; j++) {
    float *v = dbTensor.slice(0, ind[j], ind[j + 1]).contiguous().data_ptr<float>();
    for (int64_t k = 0; k < vecDim; k++) {
      auto dist = v[k] - mean[k];
      var[k] = dist * dist;
    }
  }
  cutfeat = selectDivision(var);
  //printf("cutfeat = %d  ", cutfeat);
  cutval = mean[cutfeat];
  //printf("cutval = %.2f", cutval);
  int lim1, lim2;
  planeSplit(ind, count, cutfeat, cutval, lim1, lim2);
  if (lim1 > count / 2) {
    index = lim1;
  } else if (lim2 < count / 2) {
    index = lim2;
  } else {
    index = count / 2;
  }
  if (lim1 == count || lim2 == 0) {
    index = count / 2;
  }
}

int KDTree::selectDivision(float *v) {
  int num = 0;
  size_t topind[KDTree::RAND_DIM];
  for (int64_t i = 0; i < vecDim; i++) {
    if (num < KDTree::RAND_DIM || v[i] > v[topind[num - 1]]) {
      if (num < KDTree::RAND_DIM) {
        topind[num++] = i;
      } else {
        topind[num - 1] = i;
      }

      int j = num - 1;
      while (j > 0 && v[topind[j]] > v[topind[j - 1]]) {
        std::swap(topind[j], topind[j - 1]);
        --j;
      }
    }
  }
  int rand = std::rand() % (num);
  return (int) (topind[rand]);
}

void KDTree::planeSplit(int64_t *ind, int count, int64_t cutfeat, float cutval, int &lim1, int &lim2) {
  int left = 0;
  int right = count - 1;
  for (;;) {
    auto left_data = dbTensor.slice(0, ind[left], ind[left] + 1).contiguous().data_ptr<float>();
    auto right_data = dbTensor.slice(0, ind[right], ind[right] + 1).contiguous().data_ptr<float>();
    while (left <= right && left_data[cutfeat] < cutval) ++left;
    while (left <= right && right_data[cutfeat] >= cutval) --right;
    if (left > right) break;
    std::swap(ind[left], ind[right]);
    ++left;
    --right;
  }
  lim1 = left;
  right = count - 1;
  for (;;) {
    auto left_data = dbTensor.slice(0, ind[left], ind[left] + 1).contiguous().data_ptr<float>();
    auto right_data = dbTensor.slice(0, ind[right], ind[right] + 1).contiguous().data_ptr<float>();
    while (left <= right && left_data[cutfeat] < cutval) ++left;
    while (left <= right && right_data[cutfeat] >= cutval) --right;
    if (left > right) break;
    std::swap(ind[left], ind[right]);
    ++left;
    --right;
  }

  lim2 = left;
}

void KDTree::buildTree() {
  // Free existing trees
  for (auto &root : tree_roots) {
    if (root != nullptr) {
      delete root;
      root = nullptr;
    }
  }
  // Then build
  std::vector<int64_t> idx(dbTensor.size(0));
  for (int64_t i = 0; i < static_cast<int64_t>(dbTensor.size(0)); i++) {
    idx[i] = i;
  }
  mean = new float[vecDim];
  var = new float[vecDim];

  tree_roots.resize(num_trees);
  for (uint64_t i = 0; i < tree_roots.size(); i++) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(idx.begin(), idx.end(), g);
    tree_roots[i] = divideTree(&idx[0], ntotal);
  }
  delete[] mean;
  delete[] var;
}

void KDTree::addPointToTree(NodePtr node, int64_t idx) {
  auto new_data = dbTensor.slice(0, idx, idx + 1).contiguous().data_ptr<float>();
  if (node->child1 == nullptr && node->child2 == nullptr) {
    auto leaf_data = node->data.contiguous().data_ptr<float>();
    float max_span = 0;
    int64_t div_feat = 0;
    for (size_t i = 0; i < vecDim; i++) {
      auto span = std::abs(leaf_data[i] - new_data[i]);
      if (span > max_span) {
        max_span = span;
        div_feat = i;
      }
    }
    NodePtr left = new Node();
    NodePtr right = new Node();

    if (new_data[div_feat] < leaf_data[div_feat]) {
      left->divfeat = idx;
      left->data = dbTensor.slice(0, idx, idx + 1);
      right->divfeat = node->divfeat;
      right->data = node->data;
    } else {
      left->divfeat = node->divfeat;
      left->data = node->data;
      right->divfeat = idx;
      right->data = dbTensor.slice(0, idx, idx + 1);
    }
    node->divfeat = div_feat;
    node->divval = (new_data[div_feat] + leaf_data[div_feat]) / 2;
    node->child1 = left;
    node->child2 = right;
  } else {
    if (new_data[node->divfeat] < node->divval) {
      addPointToTree(node->child1, idx);
    } else {
      addPointToTree(node->child2, idx);
    }
  }

}

void KDTree::searchLevel(ResultSet &result, const float *vec, NodePtr node, float mindist, int &checkCount,
                         int maxCheck, float epsError, Heap<BranchSt> *heap, VisitBitset &checked) {

  if (result.worstDist() < mindist) {
    return;
  }
  // if leaf node, do check and return
  if (node->child1 == nullptr && node->child2 == nullptr) {
    auto index = node->divfeat;
    if (checked.test(index) || ((checkCount >= maxCheck) && result.isFull())) {
      return;
    }
    checked.set(index);
    checkCount++;
    auto node_data = node->data.contiguous().data_ptr<float>();
    auto dist = CANDY::computeL2Distance(node_data, vec, vecDim);
    result.add(dist, index);
    return;
  }

  auto val = vec[node->divfeat];
  auto diff = val - node->divval;
  auto bestChild = (diff < 0) ? node->child1 : node->child2;
  auto otherChild = (diff < 0) ? node->child2 : node->child1;

  auto new_dist = mindist + (val - node->divval) * (val - node->divval);
  if ((new_dist * epsError < result.worstDist()) || !result.isFull()) {
    heap->push(BranchSt(otherChild, new_dist));
  }

  // recursively search to next level down
  searchLevel(result, vec, bestChild, mindist, checkCount, maxCheck, epsError, heap, checked);
}

void KDTree::get_neighbors(ResultSet &result, const float *vec, int maxCheck, float epsError) {
  BranchSt branch;
  int checkCount = 0;
  Heap<BranchSt> heap;
  VisitBitset checked(ntotal);
  for (uint64_t i = 0; i < num_trees; i++) {
    searchLevel(result, vec, tree_roots[i], 0, checkCount, maxCheck, epsError, &heap, checked);
  }

  while (heap.pop(branch) && (checkCount < maxCheck || !result.isFull())) {
    searchLevel(result, vec, branch.node, 0, checkCount, maxCheck, epsError, &heap, checked);
  }
}
}

