//
// Created by zyt on 24-10-14.
//
#include <Algorithms/kd_tree.hpp>
KDTree::KDTree(size_t dimensions): vecDim(dimensions), mean(0), var(0), lastNNZ(-1), expandStep(100), eps(0.0), checks(32){
    dbTensor = torch::zeros({0, (int64_t) vecDim});
}


void KDTree::insert(size_t id, const std::vector<float> &vec) override {
    // transform vector to tensor then insert
    torch::Tensor tensor_vec = torch::from_blob((float*)vec.data(), {1, (long)vec.size()}, torch::kFloat32).clone();
    addPoints(tensor_vec);
}

std::vector<size_t> KDTree::query(const std::vector<float> &query_vec, size_t k) const override {
    // transform the query vector to tensor
    torch::Tensor tensor_query = torch::from_blob((float*)query_vec.data(), {1, (long)query_vec.size()}, torch::kFloat32).clone();

    // saving indices and distances
    int64_t indices[k];
    float distances[k];

    // check k nearest neighbours via KNN
    int found_neighbors = knnSearch(tensor_query, indices, distances, k);

    // transform the result to vector
    std::vector<size_t> result(indices, indices + found_neighbors);
    return result;
}

void KDTree::remove(size_t id) override{
    //  TODO: prepare a remove function for deleting
}

bool KDTree::setConfig(INTELLI::ConfigMapPtr cfg) {

  vecDim = cfg->tryI64("vecDim", 768, true);
  num_trees = cfg->tryI64("numTrees", 4, true);
  tree_roots = std::vector<KDTree::NodePtr>(num_trees, nullptr);

  assert(cfg);
  dbTensor = torch::zeros({0, (int64_t) vecDim});
  lastNNZ = -1;
  expandStep = 100;
  return true;
}

bool CANDY::KdTree::setParams(FlannParam param) {
  num_trees = param.num_trees;
  printf("Best param for KdTree\n num_trees: %ld\n", num_trees);
  return true;
}

int KDTree::knnSearch(torch::Tensor &q, int64_t *idx, float *distances, int64_t aknn) {
  int count = 0;

  for (int64_t i = 0; i < q.size(0); i++) {
    FLANN::ResultSet resultSet = FLANN::ResultSet(aknn);

    auto query_data = q.slice(0, i, i + 1).contiguous().data_ptr<float>();

    getNeighbors(resultSet, query_data, checks, eps + 1);
    int64_t n = std::min(resultSet.size(), (size_t) aknn);
    resultSet.copy(idx, distances, i, n);
    //resultSet.copy(idx,distances,n);
    count += n;
  }
  return count;
}

void KDTree::getNeighbors(FLANN::ResultSet &result, const float *vec, int maxCheck, float epsError) {
  KDTree::BranchSt branch;
  int checkCount = 0;
  auto heap = FLANN::Heap<BranchSt>(ntotal);
  FLANN::VisitBitset checked;
  checked.resize(ntotal);
  for (uint64_t i = 0; i < num_trees; i++) {
    searchLevel(result, vec, tree_roots[i], 0, checkCount, maxCheck, epsError, &heap, checked);
  }

  while (heap.popMin(branch) && (checkCount < maxCheck) || !result.isFull()) {
    searchLevel(result, vec, branch.node, 0, checkCount, maxCheck, epsError, &heap, checked);
  }
}

void KDTree::searchLevel(FLANN::ResultSet &result,
                                const float *vec,
                                NodePtr node,
                                float mindist,
                                int &checkCount,
                                int maxCheck,
                                float epsError,
                                FLANN::Heap<BranchSt> *heap,
                                FLANN::VisitBitset &checked) {
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
    auto dist = faiss::fvec_L2sqr(node_data, vec, vecDim);
    result.add(dist, index);
    return;
  }

  auto val = vec[node->divfeat];
  auto diff = val - node->divval;
  auto bestChild = (diff < 0) ? node->child1 : node->child2;
  auto otherChild = (diff < 0) ? node->child2 : node->child1;

  /// TODO:not sure + IP
  auto new_dist = mindist + (val - node->divval) * (val - node->divval);
  if ((new_dist * epsError < result.worstDist()) || !result.isFull()) {
    heap->insert(BranchSt(otherChild, new_dist));
  }

  // recursively search to next level down
  searchLevel(result, vec, bestChild, mindist, checkCount, maxCheck, epsError, heap, checked);
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


void KDTree::addPointToTree(KDTree::NodePtr node, int64_t idx) {
  auto new_data = dbTensor.slice(0, idx, idx + 1).contiguous().data_ptr<float>();
  if (node->child1 == nullptr && node->child2 == nullptr) {
    auto leaf_data = node->data.contiguous().data_ptr<float>();
    float max_span = 0;
    int64_t div_feat = 0;
    for (int64_t i = 0; i < vecDim; i++) {
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

void KDTree::addPoints(torch::Tensor &t) {
  bool success = INTELLI::IntelliTensorOP::appendRowsBufferMode(&dbTensor, &t, &lastNNZ, expandStep);
  assert(success);
  ntotal += t.size(0);
  if ((ntotal - t.size(0)) * 2 < ntotal) {
    printf("starting  re-building tree\n");
    buildTree();
  } else {
    printf("Adding to existing tree\n");
    for (uint64_t i = ntotal - t.size(0); i < ntotal; i++) {
      for (uint64_t j = 0; j < num_trees; j++) {
        addPointToTree(tree_roots[j], i);
      }
    }
  }
}

void KDTree::buildTree() {
  // first free
  for (uint64_t i = 0; i < tree_roots.size(); i++) {
    if (tree_roots[i] != nullptr) {
      tree_roots[i]->~Node();
    }
  }
  // Then build
  // Create a permutation
  std::vector<int64_t> idx(dbTensor.size(0));
  for (int64_t i = 0; i < (int64_t) dbTensor.size(0); i++) {
    idx[i] = i;
  }
  mean = new float[vecDim];
  var = new float[vecDim];

  tree_roots.resize(num_trees);
  //printf("random assigning points\n");
  for (uint64_t i = 0; i < tree_roots.size(); i++) {
    // randomize the order of vectors to allow for unbiased sampling
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(idx.begin(), idx.end(), g);
    tree_roots[i] = divideTree(&idx[0], ntotal);
  }
  delete[] mean;
  delete[] var;

}


struct KDTree::Node {
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
