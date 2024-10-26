/*
* Copyright (C) 2024 by the INTELLI team
 * Created on: 2024/10/14
 * Description: [Provide description here]
 */
#ifndef KD_TREE_HPP
#define KD_TREE_HPP

#include <Algorithms/ANNSBase.hpp>
#include <Algorithms/KDTree/KDTreeUtils.hpp>
class KDTree : public ANNSBase {
public:
    typedef Node *NodePtr;
    typedef BranchStruct<NodePtr> BranchSt;
    typedef BranchSt *Branch;

private:
    uint64_t num_trees;
    float *mean;
    float *var;
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

    ~KDTree();

    bool insertTensor(const torch::Tensor &t);

    bool loadInitialTensor(const torch::Tensor &t);

    bool deleteTensor(const torch::Tensor &t, int64_t k = 1);

    bool reviseTensor(const torch::Tensor &t, const torch::Tensor &w);

    std::vector<idx_t> searchIndex(const torch::Tensor &q, int64_t k);

    std::vector<torch::Tensor> searchTensor(const torch::Tensor &q, int64_t k);

    torch::Tensor rawData();

private:
    void addPoints(torch::Tensor &t);

    int knnSearch(torch::Tensor &q, int64_t *idx, float *distances, int64_t aknn);

    NodePtr divideTree(int64_t *idx, int count);

    void buildTree();

    void addPointToTree(NodePtr node, int64_t idx);

    void get_neighbors(ResultSet &result, const float *vec, int maxCheck, float epsError);

    void searchLevel(ResultSet &result, const float *vec, NodePtr node, float mindist, int &checkCount, int maxCheck,
                     float epsError, Heap<BranchSt> *heap, VisitBitset &checked);
};

#endif // KD_TREE_HPP

