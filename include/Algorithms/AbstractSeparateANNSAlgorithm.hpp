/*
* Copyright (C) 2024 by the jjzhao
 * Created on: 2024/11/24
 * Description: [Provide description here]
 */
#ifndef CANDY_INCLUDE_ALGO_AbstractSparateANNSAlgorithm_H_
#define CANDY_INCLUDE_ALGO_AbstractSparateANNSAlgorithm_H_
#include <torch/torch.h>
#include <IO/AbstractStorageEngine.hpp>
#include <vector>

namespace CANDY_ALGO {
class AbstractSeparateANNSAlgorithm{
public:
  AbstractSeparateANNSAlgorithm() = default;
  virtual ~AbstractSeparateANNSAlgorithm() = default;
  CANDY_STORAGE::AbstractStorageEnginePtr storage_engine;
    /**
     * @brief insert a tensor
     * @param t the tensor, some index needs to be single row
     * @return bool whether the insertion is successful
     */
  virtual bool insertTensor(const torch::Tensor &t) = 0;
  /**
   * @brief search the k-NN of a query tensor, return the result rowIDs
   * @param t the tensor, allow multiple rows
   * @param k the returned neighbors
   * @return std::vector<int> the result rowIDs
   */
  virtual std::vector<torch::Tensor> searchTensor(const torch::Tensor &t, int64_t k) = 0;
  /**
   * @brief Delete tensors similar to the query tensor in the database and return the rowIDs of the deleted tensors
   * @param t the tensor, some index needs to be single row
   * @param k the number of nearest neighbors
   * @return std::vector<int> the result rowIDs to be deleted
   */
  virtual std::vector<torch::Tensor> deleteTensor(const torch::Tensor &t, int64_t k) = 0;
  virtual bool reviseTensor(const torch::Tensor &t, const torch::Tensor &w) = 0;
  /**
   * @brief search the k-NN of a query tensor, return the result tensors
   * @param t the tensor, allow multiple rows
   * @param k the returned neighbors
   * @return std::vector<int> the result vids
   */
  virtual std::vector<int64_t> findKnnTensor(const torch::Tensor &t, int64_t k) = 0;
};
}  // namespace CANDY_ALGO
#endif
