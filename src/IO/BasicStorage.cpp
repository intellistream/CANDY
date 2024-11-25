#include <IO/BasicStorage.hpp>
#include <ComputeEngine/BasicComputeEngine.hpp>

BasicStorage::BasicStorage(){

}
BasicStorage::~BasicStorage() {

}

int BasicStorage::getVid(){
  return nowVid++;
}

bool BasicStorage::insertTensorWithRawId(const torch::Tensor &vector, int rawId){
  int myvid = getVid();
  vectorPair myvectorPair;
  myvectorPair.rawId=rawId;
  myvectorPair.vector=vector;
  storageVector.insert({myvid, myvectorPair});
  return true;
}
bool BasicStorage::insertTensor(const torch::Tensor &vector){
  int myvid=getVid();
  vectorPair myvectorPair;
  myvectorPair.rawId=myvid;
  myvectorPair.vector=vector;
  storageVector.insert({myvid, myvectorPair});
  return true;
}
std::vector<int> BasicStorage::deleteTensor(std::vector<int> vids){
  std::vector<int> result;
  for(int i = 0; i < vids.size(); i++){
    auto it = storageVector.find(vids[i]);
    if(it != storageVector.end()){
      storageVector.erase(it);
      result.push_back(it -> second.rawId);
    }
  }
  return result;
}

float BasicStorage::distanceCompute(int vid1, int vid2){
    auto it1 = storageVector.find(vid1);
    auto it2 = storageVector.find(vid2);
    if(it1 != storageVector.end() && it2 != storageVector.end()){
        return computeEngine.euclidean_distance(it1 -> second.vector, it2 -> second.vector);
    } else {
        return -1;
    }
}
float BasicStorage::distanceCompute(const torch::Tensor &vector, int vid){
  return 0.0;
}
torch::Tensor BasicStorage::getVectorByVid(int vid) {
  auto it = storageVector.find(vid);
  if(it != storageVector.end()){
    return it -> second.vector;
  } else {
    std::cout<<"not find"<<std::endl;
    return torch::zeros({1, 1});
  }
}
int BasicStorage::getRawIdByVid(int vid){
  auto it = storageVector.find(vid);
  if(it != storageVector.end()){
    return it -> second.rawId;
  } else {
    return -1;
  }
}
std::string BasicStorage::display() {
  string result;
  for(auto it = storageVector.begin(); it != storageVector.end(); it++) {
    result += "vid is " + to_string(it-> first) + "\n";
    result += "rawid is " + to_string(it->second.rawId) + "\n";
    result += "embedding is " + it->second.vector.toString() + "\n";
  }
  return result;
}
std::vector<torch::Tensor> BasicStorage::getAll(){
  std::vector<torch::Tensor> tensorList;

  for (const auto& pair : storageVector) {
    tensorList.push_back(pair.second.vector);
  }

  return tensorList;
}
