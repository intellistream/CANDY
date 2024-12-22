/*
* Copyright (C) 2024 by the INTELLI team
 * Created by: Jianjun Zhao
 * Created on: 2024/12/21
 */
#include <IO/BasicStorage.hpp>
#include <ComputeEngine/BasicComputeEngine.hpp>
namespace CANDY_STORAGE {
BasicStorage::BasicStorage(){
  this -> compute_engine = std::make_shared<CANDY_COMPUTE::BasicComputeEngine>();
}
BasicStorage::~BasicStorage() {

}

int BasicStorage::getVid(){
  return nowVid ++;
}

bool BasicStorage::insertTensor(const torch::Tensor &vector){
  int myvid = getVid();
  storageVector.insert({myvid, vector});
  return true;
}

bool BasicStorage::insertTensor(const torch::Tensor &vector, int64_t &vid) {
  vid = getVid();
  storageVector.insert({vid, vector});
  return true;
}

std::vector<torch::Tensor> BasicStorage::deleteTensor(std::vector<int64_t> vids){
  std::vector<torch::Tensor> result;
  for(int i = 0; i < vids.size(); i++){
    auto it = storageVector.find(vids[i]);
    if(it != storageVector.end()){
      storageVector.erase(it);
      result.push_back(it -> second);
    }
  }
  return result;
}

bool BasicStorage::reviseTensor(const torch::Tensor &r_t, int64_t vid){
  auto it = storageVector.find(vid);
  if(it != storageVector.end()){
    it -> second = r_t;
    return true;
  } else {
    return false;
  }
}

float BasicStorage::distanceCompute(int64_t vid1, int64_t vid2){
  auto it1 = storageVector.find(vid1);
  auto it2 = storageVector.find(vid2);
  if(it1 != storageVector.end() && it2 != storageVector.end()){
    return this->compute_engine->euclidean_distance(it1 -> second, it2 -> second);
  } else {
    return -1;
  }
}
float BasicStorage::distanceCompute(const torch::Tensor &vector, int64_t vid){
  auto it = storageVector.find(vid);
  if(it != storageVector.end()){
    return this->compute_engine->euclidean_distance(it -> second, vector);
  } else {
    return -1;
  }
}
torch::Tensor BasicStorage::getVectorByVid(int64_t vid) {
  auto it = storageVector.find(vid);
  if(it != storageVector.end()){
    return it -> second;
  } else {
    std::cout<<"not find"<<std::endl;
    return torch::zeros({1, 1});
  }
}
std::vector<torch::Tensor> BasicStorage::getVectorByVids(std::vector<int64_t> vids){
  std::vector<torch::Tensor> result;
  for(int i = 0; i < vids.size(); i++){
    auto it = storageVector.find(vids[i]);
    if(it != storageVector.end()){
      result.push_back(it -> second);
    }
  }
  return result;
}

std::string BasicStorage::display() {
  string result;
  for(auto it = storageVector.begin(); it != storageVector.end(); it++) {
    result += "vid is " + to_string(it-> first) + "\n";
    result += "embedding is " + it->second.toString() + "\n";
  }
  return result;
}
std::vector<torch::Tensor> BasicStorage::getAll(){
  std::vector<torch::Tensor> tensorList;

  for (const auto& pair : storageVector) {
    tensorList.push_back(pair.second);
  }

  return tensorList;
}
}
