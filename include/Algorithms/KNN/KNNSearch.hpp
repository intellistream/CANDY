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


class KnnSearch;
typedef std::shared_ptr<KnnSearch> KnnSearchPtr;

class KnnSearch : public ANNSBase {
public:
 // Destructor
 ~KnnSearch() override = default;

 // Constructor with vector dimensions
 explicit KnnSearch(size_t dimensions);

 void reset();

 bool insertTensor(const torch::Tensor &t) override;

 bool loadInitialTensor(torch::Tensor &t) override;

 bool deleteTensor(torch::Tensor &t, int64_t k = 1) override;

 bool reviseTensor(torch::Tensor &t, torch::Tensor &w) override;

 std::vector<torch::Tensor> searchTensor(const torch::Tensor &q, int64_t k) override;

 bool resetIndexStatistics() override;

 INTELLI::ConfigMapPtr getIndexStatistics() override;

private:
 size_t dimensions;
 std::unordered_map<size_t, torch::Tensor> index;

};

#endif // CANDY_INCLUDE_ALGORITHMS_KNN_SEARCH_HPP_
