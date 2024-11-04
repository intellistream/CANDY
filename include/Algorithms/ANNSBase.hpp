/*
 * Copyright (C) 2024 by the INTELLI team
 * Created on: 2024/10/26 10:48
 * Description: ${DESCRIPTION}
 */

#ifndef ANNS_ALGORITHM_BASE_HPP
#define ANNS_ALGORITHM_BASE_HPP

#include <torch/torch.h>
#include <vector>
#include <Algorithms/AbstractANNSAlgorithm.hpp>
#include <Utils/ConfigMap.hpp>
#include <Utils/Param.hpp>


namespace CANDY_ALGO {
class ANNSBase;
typedef std::shared_ptr<ANNSBase> ANNSBasePtr;
class ANNSBase : public CANDY::AbstractANNS {
 public:
  virtual ~ANNSBase() = default;

  // Methods with default implementations
  // Methods with default implementations
  virtual void reset() override; // Logs if no specific reset
  virtual bool loadInitialTensor(torch::Tensor &t) override;

  bool startHPC() override; // Logs if no HPC setup
  bool endHPC() override; // Logs if no HPC termination
  virtual bool setConfig(INTELLI::ConfigMapPtr cfg) override; // Logs if no config setup
  bool setParams(CANDY::ParamPtr param) override; // Logs if no parameters set
  virtual bool resetIndexStatistics() override; // Logs if no statistics reset

  virtual INTELLI::ConfigMapPtr getIndexStatistics() override; // Logs if no statistics retrieval

  virtual bool insertTensor(const torch::Tensor &t) ;

  virtual bool deleteTensor(torch::Tensor &t, int64_t k) ;
  virtual bool reviseTensor(torch::Tensor &t, torch::Tensor &w);

  virtual std::vector<torch::Tensor> searchTensor(const torch::Tensor &q, int64_t k);
};
}
#endif // ANNS_ALGORITHM_BASE_HPP
