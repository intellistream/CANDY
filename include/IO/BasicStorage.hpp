/*
* Copyright (C) 2024 by the jjzhao
 * Created on: 2024/11/24
 * Description: [Provide description here]
 */
#ifndef BASIC_STORAGE_HPP
#define BASIC_STORAGE_HPP
#include <torch/torch.h>
#include <Utils/ConfigMap.hpp>
#include <map>
#include <string>
#include <vector>
#include <IO/AbstractStorageEngine.hpp>

struct vectorPair{
  torch::Tensor vector;
  int rawId;
};

class BasicStorage: public AbstractStorageEngine{
public:
  map <int, vectorPair> storageVector;
  int nowVid=0;

  BasicStorage();
  ~BasicStorage() override;
  int getVid() override;
  bool insertTensorWithRawId(const torch::Tensor &vector, int rawId) override;
  bool insertTensor(const torch::Tensor &vector) override;
  std::vector<int> deleteTensor(std::vector<int> vids) override;
  float distanceCompute(int vid1, int vid2) override;
  float distanceCompute(const torch::Tensor &vector, int vid) override;
  torch::Tensor getVectorByVid(int vid) override;
  int getRowIdByVid(int vid) override;
  std::vector<torch::Tensor> getAll() override;
  std::string display() override;
};
#endif  // BASIC_STORAGE_HPP