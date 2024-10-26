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
 * @class AbstractIndex
 * @brief The abstract class of an index approach
 */
class AbstractIndex {
protected:
  MetricType Metric = METRIC_L2;
  int64_t containerTier = 0;
public:
  bool isHPCStarted = false;
  AbstractIndex() = default;

  virtual ~AbstractIndex() = default;
  /**
  * @brief set the tier of this indexing, 0 refers the entry indexing
  * @param tie the setting of tier number
  * @note The parameter of tier idx affects nothing now, but will do something later
  */
  virtual void setTier(int64_t tie) {
    containerTier = tie;
  }
  /**
    * @brief reset this index to inited status
    */
  virtual void reset();
  /**
  * @brief set the index-specific config related to one index
  * @param cfg the config of this class, using raw class
  * @note If there is any pre-built data structures, please load it in implementing this
  * @note If there is any initial tensors to be stored, please load it after this by @ref loadInitialTensor
  * @return bool whether the configuration is successful
  */
  virtual bool setConfigClass(ConfigParser cfg);
  /**
   * @brief set the index-specfic config related to one index
   * @param cfg the config of this class
   * @note If there is any pre-built data structures, please load it in implementing this
   * @note If there is any initial tensors to be stored, please load it after this by @ref loadInitialTensor
   * @return bool whether the configuration is successful
   */
  virtual bool setConfig(ConfigParserPtr cfg);
  /**
   * @brief some extra set-ups if the index has HPC fetures
   * @return bool whether the HPC set-up is successful
   */
  virtual bool startHPC();
  /**
   * @brief insert a tensor
   * @note This is majorly an online function
   * @param t the tensor, some index need to be single row
   * @return bool whether the insertion is successful
   */
  virtual bool insertTensor(torch::Tensor &t);

  /**
  * @brief load the initial tensors of a data base, use this BEFORE @ref insertTensor
  * @note This is majorly an offline function, and may be different from @ref insertTensor for some indexes
  * @param t the tensor, some index need to be single row
  * @return bool whether the loading is successful
  */
  virtual bool loadInitialTensor(torch::Tensor &t);
  /**
   * @brief delete a tensor, also online function
   * @param t the tensor, some index needs to be single row
   * @param k the number of nearest neighbors
   * @return bool whether the deleting is successful
   */
  virtual bool deleteTensor(torch::Tensor &t, int64_t k = 1);

  /**
   * @brief revise a tensor
   * @param t the tensor to be revised
   * @param w the revised value
   * @return bool whether the revising is successful
   */
  virtual bool reviseTensor(torch::Tensor &t, torch::Tensor &w);
  /**
   * @brief search the k-NN of a query tensor, return their index
   * @param t the tensor, allow multiple rows
   * @param k the returned neighbors
   * @return std::vector<faiss::idx_t> the index, follow faiss's order
   */
  virtual std::vector<idx_t> searchIndex(torch::Tensor q, int64_t k);

  /**
   * @brief return a vector of tensors according to some index
   * @param idx the index, follow faiss's style, allow the KNN index of multiple queries
   * @param k the returned neighbors, i.e., will be the number of rows of each returned tensor
   * @return a vector of tensors, each tensor represent KNN results of one query in idx
   */
  virtual std::vector<torch::Tensor> getTensorByIndex(std::vector<idx_t> &idx, int64_t k);
  /**
    * @brief return the rawData of tensor
    * @return The raw data stored in tensor
    */
  virtual torch::Tensor rawData();
  /**
  * @brief search the k-NN of a query tensor, return the result tensors
  * @param q the tensor, allow multiple rows
  * @param k the returned neighbors
  * @return std::vector<torch::Tensor> the result tensor for each row of query
  */
  virtual std::vector<torch::Tensor> searchTensor(torch::Tensor &q, int64_t k);
  /**
    * @brief some extra termination if the index has HPC features
    * @return bool whether the HPC termination is successful
    */
  virtual bool endHPC();
  /**
   * @brief set the frozen level of online updating internal state
   * @param frozenLv the level of frozen, 0 means freeze any online update in internal state
   * @return whether the setting is successful
   */
  virtual bool setFrozenLevel(int64_t frozenLv);
  /**
  * @brief offline build phase
  * @param t the tensor for offline build
  * @note This is to generate some offline data structures, NOT load offline tensors
  * @note Please use @ref loadInitialTensor for loading initial tensors
  * @return whether the building is successful
  */
  virtual bool offlineBuild(torch::Tensor &t);
  /**
   * @brief a busy waiting for all pending operations to be done
   * @return bool, whether the waiting is actually done;
   */
  virtual bool waitPendingOperations();

  /**
  * @brief load the initial tensors of a data base along with its string objects, use this BEFORE @ref insertTensor
  * @note This is majorly an offline function, and may be different from @ref insertTensor for some indexes
  * @param t the tensor, some index need to be single row
   *  * @param strs the corresponding list of strings
  * @return bool whether the loading is successful
  */
  virtual bool loadInitialStringObject(torch::Tensor &t, std::vector<std::string> &strs);
  /**
* @brief load the initial tensors of a data base along with its string objects, use this BEFORE @ref insertTensor
* @note This is majorly an offline function, and may be different from @ref insertTensor for some indexes
* @param t the tensor, some index need to be single row
 *  * @param u64s the corresponding list of uint64_t
* @return bool whether the loading is successful
*/
  virtual bool loadInitialU64Object(torch::Tensor &t, std::vector<uint64_t> &u64s);
  /**
   * @brief insert a string object
   * @note This is majorly an online function
   * @param t the tensor, some index need to be single row
   * @param strs the corresponding list of strings
   * @return bool whether the insertion is successful
   */
  virtual bool insertStringObject(torch::Tensor &t, std::vector<std::string> &strs);
  /**
  * @brief insert a u64 object
  * @note This is majorly an online function
  * @param t the tensor, some index need to be single row
  * @param u64s the corresponding list of u64
  * @return bool whether the insertion is successful
  */
  virtual bool insertU64Object(torch::Tensor &t, std::vector<uint64_t> &u64s);
  /**
   * @brief  delete tensor along with its corresponding string object
   * @note This is majorly an online function
   * @param t the tensor, some index need to be single row
   * @param k the number of nearest neighbors
   * @return bool whether the delet is successful
   */
  virtual bool deleteStringObject(torch::Tensor &t, int64_t k = 1);
  /**
   * @brief  delete tensor along with its corresponding U64 object
   * @note This is majorly an online function
   * @param t the tensor, some index need to be single row
   * @param k the number of nearest neighbors
   * @return bool whether the delet is successful
   */
  virtual bool deleteU64Object(torch::Tensor &t, int64_t k = 1);
  /**
 * @brief search the k-NN of a query tensor, return the linked string objects
 * @param t the tensor, allow multiple rows
 * @param k the returned neighbors
 * @return std::vector<std::vector<std::string>> the result object for each row of query
 */
  virtual std::vector<std::vector<std::string>> searchStringObject(torch::Tensor &q, int64_t k);
  /**
* @brief search the k-NN of a query tensor, return the linked U64 objects
* @param t the tensor, allow multiple rows
* @param k the returned neighbors
* @return std::vector<std::vector<std::string>> the result object for each row of query
*/
  virtual std::vector<std::vector<uint64_t >> searchU64Object(torch::Tensor &q, int64_t k);
  /**
 * @brief search the k-NN of a query tensor, return the linked string objects and original tensors
 * @param t the tensor, allow multiple rows
 * @param k the returned neighbors
 * @return std::tuple<std::vector<torch::Tensor>,std::vector<std::vector<std::string>>>
 */
  virtual std::tuple<std::vector<torch::Tensor>, std::vector<std::vector<std::string>>> searchTensorAndStringObject(
      torch::Tensor &q,
      int64_t k);

  /**
  * @brief load the initial tensors and query distributions of a data base, use this BEFORE @ref insertTensor
  * @note This is majorly an offline function, and may be different from @ref insertTensor for some indexes
  * @param t the data tensor
  * @param query the example query tensor
  * @return bool whether the loading is successful
  */
  virtual bool loadInitialTensorAndQueryDistribution(torch::Tensor &t, torch::Tensor &query);
  /**
   * @brief to reset the internal statistics of this index
   * @return whether the reset is executed
   */
  virtual bool resetIndexStatistics(void);
  /**
   * @brief to get the internal statistics of this index
   * @return the statistics results in ConfigMapPtr
   */
  virtual ConfigParserPtr getIndexStatistics(void);
};

/**
 * @ingroup  CANDY_lib_bottom
 * @typedef AbstractIndexPtr
 * @brief The class to describe a shared pointer to @ref  AbstractIndex

 */
typedef std::shared_ptr<class AbstractIndex> AbstractIndexPtr;
/**
 * @ingroup  CANDY_lib_bottom
 * @def newAbstractIndex
 * @brief (Macro) To creat a new @ref  AbstractIndex shared pointer.
 */
#define newAbstractIndex std::make_shared<AbstractIndex>

#endif //CANDY_INCLUDE_ALGORITHMS_ABSTRACT_INDEX_HPP_
