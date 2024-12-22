/*
* Copyright (C) 2024 by the jjzhao
 * Created on: 2024/11/24
 * Description: [Provide description here]
 */
#ifndef BASIC_STORAGE_HPP
#define BASIC_STORAGE_HPP
#include <torch/torch.h>
#include <IO/AbstractStorageEngine.hpp>
#include <Utils/ConfigMap.hpp>
#include <map>
#include <string>
#include <vector>

namespace CANDY_STORAGE {
class BasicStorage: public AbstractStorageEngine {
public:
  map <int64_t, torch::Tensor> storageVector;
  int nowVid = 0;

  BasicStorage();
  ~BasicStorage() override;
  int getVid() override;
  bool insertTensor(const torch::Tensor &vector) override;
  bool insertTensor(const torch::Tensor &vector, int64_t &vid) override;
  std::vector<torch::Tensor> deleteTensor(std::vector<int64_t> vids) override;
  bool reviseTensor(const torch::Tensor &r_t, int64_t vid) override;
  float distanceCompute(int64_t vid1, int64_t vid2) override;
  float distanceCompute(const torch::Tensor &vector, int64_t vid) override;
  torch::Tensor getVectorByVid(int64_t vid) override;
  std::vector<torch::Tensor> getVectorByVids(std::vector<int64_t> vids) override;
  std::vector<torch::Tensor> getAll() override;
  std::string display() override;
};
}
#endif  // BASIC_STORAGE_HPP