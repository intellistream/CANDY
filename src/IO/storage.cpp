#include <../include/IO/storage.hpp>


storage::storage(){

}
storage::~storage() {

}


int storage::setVid(){
  return nowvid++;
}

bool storage::insertTensor(const torch::Tensor &vector, int rawId){
  int myvid=setVid();
  vectorPair myvectorPair;
  myvectorPair.rawId=rawId;
  myvectorPair.vector=vector;
  storageVector.insert({myvid, myvectorPair});
  return true;
}

float storage::distanceCompute(int vid1, int vid2){
  return 0.0;
}

torch::Tensor storage::getvector(int vid) {
  auto it = storageVector.find(vid);
  if(it != storageVector.end()){
    return it -> second.vector;
  } else {
    std::cout<<"not find"<<std::endl;
    return torch::zeros({1, 1});
  }
}

string storage::display() {
  string result;
  for(auto it = storageVector.begin(); it != storageVector.end(); it++) {
    result += "vid is " + to_string(it-> first) + "\n";
    result += "rawid is " + to_string(it->second.rawId) + "\n";
    result += "embedding is " + it->second.vector.toString() + "\n";
  }
  return result;
}
