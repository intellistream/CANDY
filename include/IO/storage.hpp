#include <map>
#include <vector>
#include <torch/torch.h>
#include <string>
#include <Utils/ConfigMap.hpp>
#include <Utils/Param.hpp>
#include <vector>


struct vectorPair{
  torch::Tensor vector;
  int rawId;
};

class storage{
public:
  map <int, vectorPair> storageVector;
  int nowvid=0;

  storage();
  ~storage();
  int setVid();
  bool insertTensor(const torch::Tensor &vector, int rawId);
  float distanceCompute(int vid1, int vid2);
  torch::Tensor getvector(int vid);
  string display();
};