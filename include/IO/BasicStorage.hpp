#include <torch/torch.h>
#include <Utils/ConfigMap.hpp>
#include <Utils/Param.hpp>
#include <map>
#include <string>
#include <vector>

#include "AbstractStorageEngine.hpp"

struct vectorPair{
  torch::Tensor vector;
  int rawId;
};

class BasicStorage: public AbstractStorageEngine{
public:
  map <int, vectorPair> storageVector;
  int nowVid=0;

  BasicStorage();
  ~BasicStorage();
  int getVid();
  bool insertTensorWithRawId(const torch::Tensor &vector, int rawId);
  bool insertTensor(const torch::Tensor &vector);
  vector<int> deleteTensor(vector<int> vids);
  float distanceCompute(int vid1, int vid2);
  float distanceCompute(const torch::Tensor &vector, int vid);
  torch::Tensor getVectorByVid(int vid);
  int getRowIdByVid(int vid);
  std::vector<torch::Tensor> getAll();
  string display();
};