/*
 * Copyright (C) 2024 by the INTELLI team
 * Created on: 2024/10/9
 * Description: [Provide description here]
 */

#ifndef CANDY_INCLUDE_ALGORITHMS_KNN_SEARCH_HPP_
#define CANDY_INCLUDE_ALGORITHMS_KNN_SEARCH_HPP_

#include <unordered_map>
#include <vector>
#include <Algorithms/ANNSBase.hpp>
#include <memory>
#include <torch/torch.h>


namespace CANDY_ALGO {

 class KnnSearch : public ANNSBase {
 protected:
  INTELLI::ConfigMapPtr myCfg = nullptr;
  torch::Tensor dbTensor;
  int64_t lastNNZ = 0;
  int64_t vecDim = 0, initialVolume = 1000, expandStep = 100;

 public:
  // Destructor
  ~KnnSearch() override = default;

   KnnSearch() {}
  // Constructor with vector dimensions
   KnnSearch(size_t dimensions);

   virtual bool setConfig(INTELLI::ConfigMapPtr cfg) override;

   virtual  void reset() override;

   virtual bool insertTensor(const torch::Tensor &t) override;

   virtual bool deleteTensor(torch::Tensor &t, int64_t k = 1) override;

   virtual bool reviseTensor(torch::Tensor &t, torch::Tensor &w) override;

   virtual std::vector<torch::Tensor> searchTensor(const torch::Tensor &q, int64_t k) override;


 private:
  size_t dimensions;
  std::unordered_map<size_t, torch::Tensor> index;
 };



typedef std::shared_ptr<CANDY_ALGO::KnnSearch> KnnSearchPtr;
#define newKNNIndex std::make_shared<CANDY_ALGO::KnnSearch>
}

#endif // CANDY_INCLUDE_ALGORITHMS_KNN_SEARCH_HPP_
