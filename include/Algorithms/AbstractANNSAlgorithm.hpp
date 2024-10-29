/*
 * Copyright (C) 2024 by the INTELLI team
 * Created on: 2024/10/22
 * Description: [Provide description here]
 */
#ifndef CANDY_INCLUDE_ALGORITHMS_ABSTRACT_INDEX_HPP_
#define CANDY_INCLUDE_ALGORITHMS_ABSTRACT_INDEX_HPP_
#include <Algorithms/Utils/metric_type.hpp>

#include <torch/torch.h>
#include <memory>
#include <vector>

#include <Utils/ConfigMap.hpp>
#include <Utils/Param.hpp>

namespace CANDY {
 /**
  * @ingroup CANDY_lib_bottom The main body and interfaces of library function
  * @{
  */
 /**
  * @class AbstractANNS CANDY/AbstractIndex.h
  * @brief The abstract class of an index approach
  */
 class AbstractANNS {
 protected:
  MetricType faissMetric = METRIC_L2;
  int64_t containerTier = 0;

 public:
  bool isHPCStarted = false;

  AbstractANNS();

  virtual ~AbstractANNS() = default;

  /**
   * @brief Reset this index to inited status
   */
  virtual void reset() = 0;

  /**
   * @brief Set the index-specific config related to one index
   * @param cfg the config of this class, using raw class
   * @return bool whether the configuration is successful
   */
  virtual bool setConfig(INTELLI::ConfigMapPtr cfg) = 0;

  /**
   * @brief set the params from auto-tuning
   * @param param best param
   * @return true if success
   */
  virtual bool setParams(CANDY::ParamPtr param) =0;

  /**
   * @brief Some extra set-ups if the index has HPC features
   * @return bool whether the HPC set-up is successful
   */
  virtual bool startHPC() = 0;

  /**
   * @brief Some extra termination if the index has HPC features
   * @return bool whether the HPC termination is successful
   */
  virtual bool endHPC() = 0;

  /**
   * @brief Insert a tensor
   * @note This is majorly an online function
   * @param t the tensor, some index need to be single row
   * @return bool whether the insertion is successful
   */
  virtual bool insertTensor(const torch::Tensor &t) = 0;

  /**
   * @brief Load the initial tensors of a data base, use this BEFORE insertTensor
   * @note This is majorly an offline function
   * @param t the tensor, some index need to be single row
   * @return bool whether the loading is successful
   */
  virtual bool loadInitialTensor(torch::Tensor &t) = 0;

  /**
   * @brief Delete a tensor, also online function
   * @param t the tensor, some index needs to be single row
   * @param k the number of nearest neighbors
   * @return bool whether the deleting is successful
   */
  virtual bool deleteTensor(torch::Tensor &t, int64_t k = 1) = 0;

  /**
   * @brief Revise a tensor
   * @param t the tensor to be revised
   * @param w the revised value
   * @return bool whether the revising is successful
   */
  virtual bool reviseTensor(torch::Tensor &t, torch::Tensor &w) = 0;

  /**
   * @brief search the k-NN of a query tensor, return the result tensors
   * @param t the tensor, allow multiple rows
   * @param k the returned neighbors
   * @return std::vector<torch::Tensor> the result tensor for each row of query
   */
  virtual std::vector<torch::Tensor> searchTensor(const torch::Tensor &q, int64_t k) =0;

  /**
   * @brief To reset the internal statistics of this index
   * @return bool whether the reset is executed
   */
  virtual bool resetIndexStatistics() = 0;

  /**
   * @brief To get the internal statistics of this index
   * @return the statistics results in ConfigMapPtr
   */
  virtual INTELLI::ConfigMapPtr getIndexStatistics() = 0;
 };


} // namespace CANDY


#endif //CANDY_INCLUDE_ALGORITHMS_ABSTRACT_INDEX_HPP_
