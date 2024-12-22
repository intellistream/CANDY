/*
* Copyright (C) 2024 by the INTELLI team
 * Created by: Jianjun Zhao
 * Created on: 2024/12/21
 */

#ifndef CANDY_INCLUDE_ALGO_SeparateKNNSearch_H_
#define CANDY_INCLUDE_ALGO_SeparateKNNSearch_H_
#include <Algorithms/SeparateANNSBase.hpp>
#include <Utils/ConfigMap.hpp>

namespace CANDY_ALGO {
class SeparateKNNSearch : public SeparateANNSBase {
protected:
 INTELLI::ConfigMapPtr myCfg = nullptr;
 torch::Tensor dbTensor;
 int64_t lastNNZ = 0;
 int64_t vecDim = 128, initialVolume = 1000, expandStep = 100;
 bool setConfig(INTELLI::ConfigMapPtr cfg);
public:
 SeparateKNNSearch () {}
 SeparateKNNSearch (size_t dimensions);
 ~SeparateKNNSearch() override = default;
 bool insertTensor(const torch::Tensor &t) override;
 std::vector<torch::Tensor> searchTensor(const torch::Tensor &t, int64_t k) override;
 std::vector<torch::Tensor> deleteTensor(const torch::Tensor &t, int64_t k) override;
 bool reviseTensor(const torch::Tensor &t, const torch::Tensor &w) override;
 std::vector<torch::Tensor> findKnnTensorBurst(const torch::Tensor &q, int64_t k);
private:
 size_t dimensions;
};
}

#endif  // CANDY_INCLUDE_ALGO_SeparateKNNSearch_H_