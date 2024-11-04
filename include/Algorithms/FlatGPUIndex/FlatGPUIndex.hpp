/*! \file FlatGPUIndex.h*/
//
// Created by tony on 04/01/24.
//

#ifndef CANDY_ALGO_INCLUDE_CANDY_ALGO_FlatGPUIndex_H_
#define CANDY_ALGO_INCLUDE_CANDY_ALGO_FlatGPUIndex_H_
#include "Utils/ConfigMap.hpp"
#include <memory>
#include <vector>
//#include <Utils/IntelliTensorOP.hpp>
#include "Algorithms/ANNSBase.hpp"
#include "disk_mem_buffer.h"
namespace  CANDY_ALGO {
class FlatGPUIndex;
/**
 * @ingroup  CANDY_ALGO_lib_bottom The main body and interfaces of library function
 * @{
 */
/**
 * @class FlatGPUIndex CANDY_ALGO/FlatGPUIndex.h
 * @brief Similar to @ref FlatAMMIPObjectIndex, but able to run on GPU for DCO
 * @note currently single thread
 * @note config parameters
 * - vecDim, the dimension of vectors, default 768, I64
 * - initialVolume, the initial volume of inline database tensor, default 1000, I64
 * - memBufferSize, the size of memory buffer, in rows of vectors, MUST larger than designed data size, default 1000, I64
 * - sketchSize, the sketch size of amm, default 10, I64
 * - DCOBatchSize, the batch size of internal distance comparison operation (DCO), default equal to memBufferSize, I64
 * - cudaDevice, the cuda device for DCO, default -1 (none), I64
 * @warning please run the benchmark/scripts/setupSPDK/drawTogether.py at generation path before using SSD
 */
class FlatGPUIndex : public ANNSBase {
 protected:
  INTELLI::ConfigMapPtr myCfg = nullptr;
  torch::Tensor dbTensor, objTensor;
  PlainMemBufferTU dmBuffer;
  int64_t ammType = 0;
  int64_t sketchSize = 10;
  int64_t DCOBatchSize = -1;
  int64_t memBufferSize = 1000;
  int64_t vecDim = 768;
  int64_t cudaDevice = -1;

  // Main function to process batches and find top_k closest vectors
  std::vector<int64_t> findTopKClosest(const torch::Tensor &query, int64_t top_k, int64_t batch_size);
  // torch::Tensor myMMInline(torch::Tensor &a, torch::Tensor &b, int64_t ss = 10);
  /**
  * @brief return a vector of tensors according to some index
  * @param idx the index, follow faiss's style, allow the KNN index of multiple queries
  * @param k the returned neighbors, i.e., will be the number of rows of each returned tensor
  * @return a vector of tensors, each tensor represent KNN results of one query in idx
  */
  virtual std::vector<torch::Tensor> getTensorByStdIdx(std::vector<int64_t> &idx, int64_t k);
  /**
   * @brief the distance function pointer member
   * @note will select largest distance during the following sorting, please convert if your distance is 'minimal'
   * @param db The data base tensor, sized [n*vecDim] to be scanned
   * @param query The query tensor, sized [q*vecDim] to be scanned
   * @param cudaDev The id of cuda device, -1 means no cuda
   * @param idx the pointer to index
   * @return The distance tensor, must sized [q*n] and remain in cpu
   */
  torch::Tensor (*distanceFunc)(torch::Tensor db, torch::Tensor query, int64_t cudaDev, FlatGPUIndex *idx);
  /**
   * @brief the distance function of inner product
   * @param db The data base tensor, sized [n*vecDim] to be scanned
   * @param query The query tensor, sized [q*vecDim] to be scanned
   * @param cudaDev The id of cuda device, -1 means no cuda
   * @param idx the pointer to index
   * @return The distance tensor, must sized [q*n], will in GPU if cuda is valid
   */
  static torch::Tensor distanceIP(torch::Tensor db, torch::Tensor query, int64_t cudaDev, FlatGPUIndex *idx);
  /**
   * @brief the distance function of L2
   * @param db The data base tensor, sized [n*vecDim] to be scanned
   * @param query The query tensor, sized [q*vecDim] to be scanned
   * @param cudaDev The id of cuda device, -1 means no cuda
   * @param idx the pointer to index
   * @return The distance tensor, must sized [q*n], will in GPU if cuda is valid
   */
  static torch::Tensor distanceL2(torch::Tensor db, torch::Tensor query, int64_t cudaDev, FlatGPUIndex *idx);
  // std::vector<faiss::idx_t> knnInline(torch::Tensor &query, int64_t k, int64_t distanceBatch = -1);
 public:
  FlatGPUIndex() {

  }

  ~FlatGPUIndex() override = default;
  int64_t gpuComputingUs = 0;
  int64_t gpuCommunicationUs = 0;

  virtual bool setConfig(INTELLI::ConfigMapPtr cfg) override;


  virtual bool insertTensor(const torch::Tensor &t) override;

  virtual bool deleteTensor(torch::Tensor &t, int64_t k = 1) override;

  virtual bool reviseTensor(torch::Tensor &t, torch::Tensor &w) override;

  virtual std::vector<torch::Tensor> searchTensor(const torch::Tensor &q, int64_t k) override;

  /**
   * @brief return the size of ingested tensors
   * @return
   */
  virtual int64_t size() {
    return dmBuffer.size();
  }
  /**
   * @brief insert a string object
   * @note This is majorly an online function
   * @param t the tensor, some index need to be single row
   * @param strs the corresponding list of strings
   * @return bool whether the insertion is successful
   */
  // virtual bool insertStringObject(torch::Tensor &t, std::vector<std::string> &strs);

  /**
   * @brief  delete tensor along with its corresponding string object
   * @note This is majorly an online function
   * @param t the tensor, some index need to be single row
   * @param k the number of nearest neighbors
   * @return bool whether the delet is successful
   */
  //virtual bool deleteStringObject(torch::Tensor &t, int64_t k = 1);

  /**
 * @brief search the k-NN of a query tensor, return the linked string objects
 * @param t the tensor, allow multiple rows
 * @param k the returned neighbors
 * @return std::vector<std::vector<std::string>> the result object for each row of query
 */
  // virtual std::vector<std::vector<std::string>> searchStringObject(torch::Tensor &q, int64_t k);

  /**
   * @brief to reset the internal statistics of this index
   * @return whether the reset is executed
   */
  virtual bool resetIndexStatistics(void) override;
  /**
   * @brief to get the internal statistics of this index
   * @return the statistics results in ConfigMapPtr
   */
  virtual INTELLI::ConfigMapPtr getIndexStatistics(void) override;
};

/**
 * @ingroup  CANDY_ALGO_lib_bottom
 * @typedef FlatGPUIndexPtr
 * @brief The class to describe a shared pointer to @ref  FlatGPUIndex

 */
typedef std::shared_ptr<class CANDY_ALGO::FlatGPUIndex> FlatGPUIndexPtr;
/**
 * @ingroup  CANDY_ALGO_lib_bottom
 * @def newFlatGPUIndex
 * @brief (Macro) To creat a new @ref  FlatGPUIndex shared pointer.
 */
#define newFlatGPUIndex std::make_shared<CANDY_ALGO::FlatGPUIndex>
}
/**
 * @}
 */
#endif //INTELLISTREAM_INCLUDE_CPPALGOS_ABSTRACTCPPALGO_H_
