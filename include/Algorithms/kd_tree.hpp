//
// Created by zyt on 24-10-14.
//

#ifndef INTELLISTREAM_SRC_ALGORITHMS_KD_TREE_HPP_
#define INTELLISTREAM_SRC_ALGORITHMS_KD_TREE_HPP_

#include <faiss/utils/distances.h>
#include <flann/flann.hpp>
#include <torch/torch.h>

#include <vector>
#include <algorithm>
#include <random>

#include "search_algorithm.hpp"



class KDTree : public SearchAlgorithm{
    typedef Node *NodePtr;
    typedef FLANN::BranchStruct<NodePtr> BranchSt;
    typedef BranchSt *Branch;

    /// Number of randomized trees that are used in forest
    uint64_t num_trees;

    float *mean;
    float *var;

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
    struct Node;
    KDTree(size_t dimensions);
 // Insert a vector into the k-NN index
    void insert(size_t id, const std::vector<float> &vec) override;

 // Query k nearest neighbors (returns vector of IDs)
    std::vector<size_t> query(const std::vector<float> &query_vec, size_t k) const override;

 // Remove a vector from the k-NN index
    void remove(size_t id) override;

  /**
  * @brief set the index-specific config related to one index
  * @param cfg the config of this class
  * @return bool whether the configuration is successful
  */
  bool setConfig(INTELLI::ConfigMapPtr cfg);



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
  void addPoints(torch::Tensor &t);


  /**
   * @brief perform knn-search on the kdTree structure
   * @param q query data to be searched
   * @param idx result vectors indices
   * @param distances result vectors' distances with query
   * @param aknn number of approximate neighbors
   * @return number of results obtained
   */
  int knnSearch(torch::Tensor &q, int64_t *idx, float *distances, int64_t aknn) ;


  /**
   * @brief set the params from auto-tuning
   * @param param best param
   * @return true if success
   */
  bool setParams(FlannParam param) ;


  /**
   * @brief
   * @param result
   * @param vec
   * @param maxCheck
   * @param epsError
   */
  void getNeighbors(FLANN::ResultSet &result, const float *vec, int maxCheck, float epsError);


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
  void searchLevel(FLANN::ResultSet &result,
                   const float *vec,
                   NodePtr node,
                   float mindist,
                   int &checkCount,
                   int maxCheck,
                   float epsError,
                   FLANN::Heap<BranchSt> *heap,
                   FLANN::VisitBitset &checked);


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
  NodePtr divideTree(int64_t *idx, int count);



  /**
   * @brief choose which feature to use to subdivide this subset of vectors by randomly choosing those with highest variance
   * @param ind index of this vector
   * @param count number of vectors in this sublist
   * @param index index where the sublist split
   * @param cutfeat index of highest variance as cut feature
   * @param cutval value of highest variance
   */
  void meanSplit(int64_t *ind, int count, int64_t &index, int64_t &cutfeat, float &cutval);



  /**
   * @brief select top RAND_DIM largest values from vector and return index of one of them at random
   * @param v values of variance
   * @return the index of randomly chosen highest variance
   */
  int selectDivision(float *v);



  /**
   * @brief subdivide the lists by a plane perpendicular on axe corresponding to the cutfeat dimension at cutval position
   * @param ind index of the list
   * @param count count of the list
   * @param cutfeat the chosen feature
   * @param cutval the threshold value to be compared
   * @param lim1 split index candidate for meansplit
   * @param lim2 split index candidate for meansplit
   */
  void planeSplit(int64_t *ind, int count, int64_t cutfeat, float cutval, int &lim1, int &lim2);



};
#endif //INTELLISTREAM_SRC_ALGORITHMS_KD_TREE_HPP_