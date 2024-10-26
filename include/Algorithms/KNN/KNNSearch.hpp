/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/9
 * Description: [Provide description here]
 */
#ifndef INTELLISTREAM_SRC_ALGORITHMS_KNN_SEARCH_HPP_
#define INTELLISTREAM_SRC_ALGORITHMS_KNN_SEARCH_HPP_

#include <unordered_map>
#include <vector>

#include <Algorithms/ANNSBase.hpp>

class KnnSearch : public ANNSBase {
public:
 // Destructor
 ~KnnSearch() override = default;

 // Constructor with vector dimensions
 explicit KnnSearch(size_t dimensions);

 // Override functions from ANNSBase
 void reset() override;

 bool setConfig(INTELLI::ConfigMapPtr cfg) override;

 bool startHPC() override;

 bool endHPC() override;

 bool insertTensor(torch::Tensor &t) override;

 bool loadInitialTensor(torch::Tensor &t) override;

 bool deleteTensor(torch::Tensor &t, int64_t k = 1) override;

 bool reviseTensor(torch::Tensor &t, torch::Tensor &w) override;

 bool resetIndexStatistics() override;

 INTELLI::ConfigMapPtr getIndexStatistics() override;

 // Additional function to find k-nearest neighbors
 std::vector<size_t> findKNearestNeighbors(const std::vector<float> &query, size_t k) const;

private:
 size_t dimensions;
 std::unordered_map<size_t, std::vector<float> > index;

 // Helper function to calculate Euclidean distance
 float calculate_distance(const std::vector<float> &vec1, const std::vector<float> &vec2) const;
};

#endif //INTELLISTREAM_SRC_ALGORITHMS_KNN_SEARCH_HPP_
