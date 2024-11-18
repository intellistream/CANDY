/*
 * Copyright (C) 2024 by the INTELLI team
 * Created on: 24-11-5 下午2:49
 * Description: ${DESCRIPTION}
 */

#include <torch/torch.h>
#include <Algorithms/LSH/LSHSearch.hpp>
#include <algorithm>
#include <iostream>
#include <random>
#include <unordered_map>

namespace CANDY_ALGO {

// Set configuration
bool LSHSearch::setConfig(INTELLI::ConfigMapPtr cfg) {
  // ANNSBase::setConfig(cfg);
  if (cfg == nullptr)
    return false;

  Dimensions = cfg->tryI64("vecDim", 768, true);
  NumofHyperplanes = cfg->tryI64("numberOfHyperplanes", 10, true);
  //printf("NumofHyperplanes = %zu\n",NumofHyperplanes);

  // Generate random hyperplanes
  GenerateRandomHyperplanes(NumofHyperplanes);
  return true;
}

// Reset the LSH data structure
void LSHSearch::reset() {
  Index.clear();
  GlobalIndexCounter = 0;
}

// Insert tensor into LSH
bool LSHSearch::insertTensor(const torch::Tensor& t) {
  for (int64_t i = 0; i < t.size(0); ++i) {
    auto row = t[i];
    int64_t id = GlobalIndexCounter++;
    std::string hashValue = HashFunction(row);
    Index[hashValue][id] = row;
  }
  return true;
}

// Delete tensor from LSH
bool LSHSearch::deleteTensor(torch::Tensor& t, int64_t k) {

  auto results = searchTensor(t, k);

  for (int64_t i = 0; i < t.size(0); ++i) {

    for (int64_t j = 0; j < k; ++j) {
      int64_t id = results[i][j].item<int64_t>();

      if (idToBucket.count(id) > 0) {
        std::string bucket = idToBucket[id];

        if (Index.count(bucket) > 0) {
          auto& BucketMap = Index[bucket];

          if (BucketMap.count(id) > 0) {
            BucketMap.erase(id);
            cout << "Delete: " << j <<endl;
          }
        }
      }
    }
  }

  return true;
}


// Revise tensor (modify its value)
bool LSHSearch::reviseTensor(torch::Tensor& t, torch::Tensor& w) {
  if (t.size(0) > w.size(0) || t.size(1) != w.size(1)) {
    return false;
  }

  t = t.contiguous();
  w = w.contiguous();

  for (int64_t i = 0; i < t.size(0); ++i) {
    auto Row = t[i];
    std::string Bucket = HashFunction(Row);

    if (Index.count(Bucket) > 0) {
      auto& BucketMap = Index[Bucket];

      for (auto it = BucketMap.begin(); it != BucketMap.end(); ++it) {
        auto TensorId = it->first;
        auto Tensor = it->second;

        if (torch::equal(Tensor, Row)) {
          BucketMap.erase(it);
          std::string NewBucket = HashFunction(w[i]);
          Index[NewBucket][TensorId] = w[i];
          break;
        }
      }
    }
  }
  return true;
}


// Search for the k nearest neighbors of tensor q

std::vector<torch::Tensor> LSHSearch::searchTensor(const torch::Tensor& q, int64_t k) {
  std::vector<torch::Tensor> Results;
  idToBucket.clear();

  for (int64_t i = 0; i < q.size(0); ++i) {
    auto Row = q[i];
    std::string Bucket = HashFunction(Row);

    std::map<float, std::vector<std::string>> rowNearbyBuckets;

    // 遍历所有桶计算哈希距离
    for (const auto& [otherBucket, _] : Index) {
      float dist = static_cast<float>(HammingDistance(Bucket, otherBucket));
      rowNearbyBuckets[dist].push_back(otherBucket);
    }

    std::vector<std::pair<float, int64_t>> Distances; // 存储距离和 ID
    for (const auto& [dist, buckets] : rowNearbyBuckets) {
      for (const auto& nBucket : buckets) {
        if (Index.count(nBucket) > 0) {
          auto& BucketMap = Index[nBucket];
          for (const auto& [TensorId, Tensor] : BucketMap) {
            float Distance = (Tensor - Row).norm().item<float>();
            Distances.emplace_back(Distance, TensorId);
            idToBucket[TensorId] = nBucket; // 绑定 ID 和桶号
          }
        }
      }
      if (Distances.size() >= k)  break;
    }

    std::sort(Distances.begin(), Distances.end());
    if (Distances.size() > k) {
      Distances.resize(k);
    }

    torch::Tensor Tensor = torch::empty({static_cast<long>(Distances.size())},
                                        torch::dtype(torch::kLong));
    for (size_t j = 0; j < Distances.size(); ++j) {
      Tensor[j] = static_cast<long>(Distances[j].second);
    }
    Results.push_back(Tensor);
  }

  return Results;
}



// Generate random hyperplanes for hashing
void LSHSearch::GenerateRandomHyperplanes(size_t NumPlanes) {
  RandomHyperplanes.resize(NumPlanes);
  for (size_t i = 0; i < NumPlanes; ++i) {
    RandomHyperplanes[i] =
        torch::empty({static_cast<long>(Dimensions)}).uniform_(-1, 1);
    RandomHyperplanes[i] = RandomHyperplanes[i] / RandomHyperplanes[i].norm();
  }
}

// Hash function to map a tensor to a bucket (returns string hash)
std::string LSHSearch::HashFunction(const torch::Tensor& t) {
  std::string hashValue = "";

  for (const auto& plane : RandomHyperplanes) {
    float dotProduct = torch::dot(t, plane).item<float>();
    hashValue += (dotProduct > 0.001) ? '1' : '0';
  }

  return hashValue;
}

// Hamming distance between two binary strings
int LSHSearch::HammingDistance(const std::string& str1,
                               const std::string& str2) {
  int dist = 0;
  for (size_t i = 0; i < str1.size(); ++i) {
    if (str1[i] != str2[i]) {
      dist++;
    }
  }
  return dist;
}

}  // namespace CANDY_ALGO
