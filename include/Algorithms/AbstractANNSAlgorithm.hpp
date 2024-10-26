/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Ziao Wang
 * Created on: 2024/10/22
 * Description: [Provide description here]
 */
#ifndef CANDY_INCLUDE_ALGORITHMS_ABSTRACT_INDEX_HPP_
#define CANDY_INCLUDE_ALGORITHMS_ABSTRACT_INDEX_HPP_
#include <Utils/config_parser.hpp>
#include <Algorithms/Utils/metric_type.hpp>

#include <torch/torch.h>
#include <memory>
#include <vector>

/**
 * @class AbstractANNSAlgorithm
 * @brief The abstract class for an ANNS index approach
 */
class AbstractANNSAlgorithm {
protected:
  MetricType metric = METRIC_L2;
  int64_t containerTier = 0;

public:
  bool isHPCStarted = false;

  AbstractANNSAlgorithm() = default;
  virtual ~AbstractANNSAlgorithm() = default;

  /**
   * @brief Set the tier of this indexing, 0 refers to the entry indexing
   * @param tier The tier number to set
   */
  virtual void setTier(int64_t tier) {
    containerTier = tier;
  }

  /**
   * @brief Reset this index to the initial state
   */
  virtual void reset() = 0;

  /**
   * @brief Set the index-specific configuration
   * @param cfg Configuration for the index
   * @return bool Whether the configuration is successful
   */
  virtual bool setConfigClass(const ConfigParser& cfg) = 0;

  /**
   * @brief Set the index-specific configuration using a pointer
   * @param cfg Pointer to the configuration
   * @return bool Whether the configuration is successful
   */
  virtual bool setConfig(const ConfigParserPtr& cfg) = 0;

  /**
   * @brief Perform extra setup if the index has HPC features
   * @return bool Whether the HPC setup is successful
   */
  virtual bool startHPC() = 0;

  /**
   * @brief Insert a tensor into the index
   * @param t The tensor to insert (some indices require a single row)
   * @return bool Whether the insertion is successful
   */
  virtual bool insertTensor(const torch::Tensor& t) = 0;

  /**
   * @brief Load initial tensors for the index
   * @param t The tensor to load (some indices require a single row)
   * @return bool Whether the loading is successful
   */
  virtual bool loadInitialTensor(const torch::Tensor& t) = 0;

  /**
   * @brief Delete a tensor from the index
   * @param t The tensor to delete (some indices require a single row)
   * @param k The number of nearest neighbors to consider
   * @return bool Whether the deletion is successful
   */
  virtual bool deleteTensor(const torch::Tensor& t, int64_t k = 1) = 0;

  /**
   * @brief Revise a tensor in the index
   * @param t The tensor to revise
   * @param w The revised value
   * @return bool Whether the revision is successful
   */
  virtual bool reviseTensor(const torch::Tensor& t, const torch::Tensor& w) = 0;

  /**
   * @brief Search for the k-nearest neighbors of a query tensor
   * @param q The query tensor (multiple rows allowed)
   * @param k The number of neighbors to return
   * @return std::vector<idx_t> The indices of the neighbors
   */
  virtual std::vector<idx_t> searchIndex(const torch::Tensor& q, int64_t k) = 0;

  /**
   * @brief Retrieve tensors by their indices
   * @param idx The indices of the tensors to retrieve
   * @param k The number of neighbors (i.e., number of rows per tensor)
   * @return std::vector<torch::Tensor> A vector of tensors representing the results
   */
  virtual std::vector<torch::Tensor> getTensorByIndex(const std::vector<idx_t>& idx, int64_t k) = 0;

  /**
   * @brief Get the raw data stored in the index as a tensor
   * @return torch::Tensor The raw data tensor
   */
  virtual torch::Tensor rawData() = 0;

  /**
   * @brief Search for the k-nearest neighbors of a query tensor, returning the result tensors
   * @param q The query tensor (multiple rows allowed)
   * @param k The number of neighbors to return
   * @return std::vector<torch::Tensor> The result tensors for each row of the query
   */
  virtual std::vector<torch::Tensor> searchTensor(const torch::Tensor& q, int64_t k) = 0;

  /**
   * @brief Terminate any HPC features if enabled
   * @return bool Whether the HPC termination is successful
   */
  virtual bool endHPC() = 0;

  /**
   * @brief Set the frozen level for online updating of internal state
   * @param frozenLv The level of frozen state (0 means freeze any online update)
   * @return bool Whether the setting is successful
   */
  virtual bool setFrozenLevel(int64_t frozenLv) = 0;

  /**
   * @brief Perform offline building of the index
   * @param t The tensor used for offline building
   * @return bool Whether the building is successful
   */
  virtual bool offlineBuild(const torch::Tensor& t) = 0;

  /**
   * @brief Wait for all pending operations to complete
   * @return bool Whether all operations are completed
   */
  virtual bool waitPendingOperations() = 0;

  /**
   * @brief Load the initial tensors and their corresponding string objects
   * @param t The tensor to load
   * @param strs The list of corresponding strings
   * @return bool Whether the loading is successful
   */
  virtual bool loadInitialStringObject(const torch::Tensor& t, const std::vector<std::string>& strs) = 0;

  /**
   * @brief Insert a tensor and its corresponding string objects
   * @param t The tensor to insert
   * @param strs The list of corresponding strings
   * @return bool Whether the insertion is successful
   */
  virtual bool insertStringObject(const torch::Tensor& t, const std::vector<std::string>& strs) = 0;

  /**
   * @brief Delete a tensor and its corresponding string objects
   * @param t The tensor to delete
   * @param k The number of neighbors to consider
   * @return bool Whether the deletion is successful
   */
  virtual bool deleteStringObject(const torch::Tensor& t, int64_t k = 1) = 0;

  /**
   * @brief Delete a tensor and its corresponding uint64_t objects
   * @param t The tensor to delete
   * @param k The number of neighbors to consider
   * @return bool Whether the deletion is successful
   */
  virtual bool deleteU64Object(const torch::Tensor& t, int64_t k = 1) = 0;

  /**
   * @brief Search for k-nearest neighbors and return the corresponding string objects
   * @param q The query tensor (multiple rows allowed)
   * @param k The number of neighbors to return
   * @return std::vector<std::vector<std::string>> The result for each row of the query
   */
  virtual std::vector<std::vector<std::string>> searchStringObject(const torch::Tensor& q, int64_t k) = 0;

  /**
   * @brief Search for k-nearest neighbors and return the corresponding uint64_t objects
   * @param q The query tensor (multiple rows allowed)
   * @param k The number of neighbors to return
   * @return std::vector<std::vector<uint64_t>> The result for each row of the query
   */
  virtual std::vector<std::vector<uint64_t>> searchU64Object(const torch::Tensor& q, int64_t k) = 0;

  /**
   * @brief Search for k-nearest neighbors, returning both the tensors and linked string objects
   * @param q The query tensor (multiple rows allowed)
   * @param k The number of neighbors to return
   * @return std::tuple<std::vector<torch::Tensor>, std::vector<std::vector<std::string>>> The result as a tuple
   */
  virtual std::tuple<std::vector<torch::Tensor>, std::vector<std::vector<std::string>>> searchTensorAndStringObject(
      const torch::Tensor& q, int64_t k) = 0;

  /**
   * @brief Load the initial tensors and query distributions
   * @param t The data tensor
   * @param query The example query tensor
   * @return bool Whether the loading is successful
   */
  virtual bool loadInitialTensorAndQueryDistribution(const torch::Tensor& t, const torch::Tensor& query) = 0;

  /**
   * @brief Reset the internal statistics of this index
   * @return bool Whether the reset is successful
   */
  virtual bool resetIndexStatistics() = 0;

  /**
   * @brief Get the internal statistics of this index
   * @return ConfigParserPtr The statistics results
   */
  virtual ConfigParserPtr getIndexStatistics() = 0;
};

/**
 * @typedef AbstractIndexPtr
 * @brief Shared pointer to an AbstractIndex
 */
typedef std::shared_ptr<AbstractANNSAlgorithm> AbstractIndexPtr;

/**
 * @def newAbstractIndex
 * @brief Macro to create a new shared pointer to AbstractIndex
 */
#define newAbstractIndex std::make_shared<AbstractANNSAlgorithm>


#endif //CANDY_INCLUDE_ALGORITHMS_ABSTRACT_INDEX_HPP_
