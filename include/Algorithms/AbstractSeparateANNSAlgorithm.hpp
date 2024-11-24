/*
* Copyright (C) 2024 by the jjzhao
 * Created on: 2024/11/24
 * Description: [Provide description here]
 */
#ifndef CANDY_INCLUDE_STORAGE_AbstractSparateANNSAlgorithm_H_
#define CANDY_INCLUDE_STORAGE_AbstractSparateANNSAlgorithm_H_
#include <torch/torch.h>
#include <vector>

#include "IO/AbstractStorageEngine.hpp"

class AbstractSeparateANNSAlgorithm{
public:
  AbstractStorageEngine storage_engine;
  AbstractSeparateANNSAlgorithm() = default;
  virtual ~AbstractSeparateANNSAlgorithm() = default;
  /**
   * @brief search the k-NN of a query tensor, return the result rowIDs
   * @param t the tensor, allow multiple rows
   * @param k the returned neighbors
   * @return std::vector<int> the result rowIDs
   */
  virtual std::vector<int> searchTensor(const torch::Tensor &t, int64_t k) = 0;
  /**
   * @brief Delete tensors similar to the query tensor in the database and return the rowIDs of the deleted tensors
   * @param t the tensor, some index needs to be single row
   * @param k the number of nearest neighbors
   * @return std::vector<int> the result rowIDs to be deleted
   */
  virtual std::vector<int> deleteTensor(const torch::Tensor &t, int64_t k) = 0;
  /**
   * @brief search the k-NN of a query tensor, return the result tensors
   * @param t the tensor, allow multiple rows
   * @param k the returned neighbors
   * @return std::vector<int> the result vids
   */
  virtual std::vector<int> findKnnTensor(const torch::Tensor &t, int64_t k) = 0;
};
#endif  //CANDY_INCLUDE_STORAGE_AbstractSparateANNSAlgorithm_H_
