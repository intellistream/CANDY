/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Ziao Wang
 * Created on: 2024/11/18
 * Description: [Provide description here]
 */

#ifndef CANDY_INCLUDE_ALGORITHMS_SONG_SONG_HPP
#define CANDY_INCLUDE_ALGORITHMS_SONG_SONG_HPP

#include <Utils/ConfigMap.hpp>
#include <Algorithms/ANNSBase.hpp>
#include <memory>
#include <vector>
#include "config.hpp"
#include "data.hpp"
#include "kernelgraph.cuh"

namespace CANDY_ALGO {
class SONG;

class SONG final : public ANNSBase {
 protected:
  INTELLI::ConfigMapPtr myCfg = nullptr;
  torch::Tensor dbTensor, objTensor;
  int64_t vecDim = 768;
  int64_t vecVolume = 1000;
  int64_t idx = 0;
  MetricType Metric = METRIC_L2;
  std::unique_ptr<SONG_KERNEL::Data> data = nullptr;
  std::unique_ptr<SONG_KERNEL::GraphWrapper> graph = nullptr;

  /**
   * @brief convert a query tensor to a vector of pairs
   * @param[in] t the query tensor
   * @param[out] res the result vector
   */
  static void convertTensorToVectorPair(torch::Tensor &t,
                                        std::vector<std::pair<int,SONG_KERNEL::value_t>> &res);

  /**
   * @brief convert a batch of query tensors to a batch of vectors of pairs
   * @param[in] ts the query tensors
   * @param[out] res the result vector
   */
  static void convertTensorToVectorPairBatch(torch::Tensor &ts,
                                             std::vector<std::vector<std::pair<int,SONG_KERNEL::value_t>>> &res);

 public:
  SONG() = default;

  ~SONG() override = default;

  int64_t gpuComputingUs = 0;
  int64_t gpuCommunicationUs = 0;

  bool setConfig(INTELLI::ConfigMapPtr cfg) override;

  bool insertTensor(const torch::Tensor &t) override;

  bool deleteTensor(torch::Tensor &t, int64_t k = 1) override;

  bool reviseTensor(torch::Tensor &t, torch::Tensor &w) override;

  std::vector<torch::Tensor> searchTensor(const torch::Tensor &q, int64_t k) override;

   [[nodiscard]] int64_t size() const {
    return idx;
  }

   bool resetIndexStatistics() override;

   INTELLI::ConfigMapPtr getIndexStatistics() override;
};

/**
 * @ingroup  CANDY_lib_bottom
 * @typedef SONGPtr
 * @brief The class to describe a shared pointer to @ref  SONG

 */
typedef std::shared_ptr<SONG> SONGPtr;
}

#endif //CANDY_INCLUDE_CANDY_SONG_HPP
