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
  NumofHyperplanes = cfg->tryI64("numberOfHyperplanes", 6, true);
  lastNNZ = -1;

  // Generate random hyperplanes
  //GenerateRandomHyperplanes(NumofHyperplanes);
  GenerateGaussianHyperplanes(NumofHyperplanes);
  return true;
}

// Reset the LSH data structure
void LSHSearch::reset() {
  Index.clear();
  idToBucket.clear();
  lastNNZ = -1;
}

// Insert tensor into LSH
bool LSHSearch::insertTensor(const torch::Tensor& t) {
  for (int64_t i = 0; i < t.size(0); ++i) {
    auto row = t[i];
    int64_t id = ++lastNNZ;
    std::string hashValue = HashFunction(row);
    Index[hashValue][id] = row;
    idToBucket[id] = hashValue;
  }
  return true;
}

// Delete tensor from LSH
bool LSHSearch::deleteTensor(torch::Tensor& t, int64_t k) {

  auto results = searchTensor(t, k);
  std::set<int64_t> idxToDeleteSet;

  // Record the IDs to be deleted and deduplicate
  for (int64_t i = 0; i < t.size(0); ++i) {
    for (int64_t j = 0; j < k; ++j) {
      int64_t id = results[i][j].item<int64_t>();
      idxToDeleteSet.insert(id);
    }
  }

  std::vector<int64_t> idxToDelete(idxToDeleteSet.begin(), idxToDeleteSet.end());
  // Sort in reverse order
  std::sort(idxToDelete.begin(), idxToDelete.end(), std::greater<int64_t>());

  for (const auto& id : idxToDelete) {
    if (idToBucket.count(id) > 0) {
      std::string bucket = idToBucket[id];

      if (Index.count(bucket) > 0) {
        auto& BucketMap = Index[bucket];

        if (BucketMap.count(id) > 0) {
          int64_t lastId = lastNNZ;

          // The difference is in the last element and the other
          if (id != lastId) {

            std::string lastBucket = idToBucket[lastId];

            auto& lastBucketMap = Index[lastBucket];
            auto lastTensor = lastBucketMap[lastId];

            lastBucketMap.erase(lastId);
            lastBucketMap[id] = lastTensor;

            idToBucket[id] = lastBucket;
          }

          idToBucket.erase(lastId);
          BucketMap.erase(id);

          --lastNNZ;
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

  for (int64_t i = 0; i < q.size(0); ++i) {
    auto Row = q[i];
    std::string Bucket = HashFunction(Row);

    // Record the Hammingdistance of all buckets from the current bucket
    std::map<float, std::vector<std::string>> rowNearbyBuckets;

    for (const auto& [otherBucket, _] : Index) {
      float dist = static_cast<float>(HammingDistance(Bucket, otherBucket));
      rowNearbyBuckets[dist].push_back(otherBucket);
    }

    std::vector<std::pair<float, int64_t>> Distances;
    for (const auto& [dist, buckets] : rowNearbyBuckets) {
      for (const auto& nBucket : buckets) {
        if (Index.count(nBucket) > 0) {
          auto& BucketMap = Index[nBucket];
          for (const auto& [TensorId, Tensor] : BucketMap) {
            float Distance = (Tensor - Row).norm().item<float>();
            Distances.emplace_back(Distance, TensorId);
          }
        }
      }
      if (Distances.size() >= k)  break;
    }

    std::sort(Distances.begin(), Distances.end());
    if (Distances.size() > k) {
      Distances.resize(k);
    }

  //  for (const auto& distPair : Distances) {
  //    float distance = distPair.first;
  //    int64_t id = distPair.second;
  //    std::cout << "Distance: " << distance << ", ID: " << id << std::endl;
  //  }

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

void LSHSearch::GenerateGaussianHyperplanes(size_t NumPlanes) {
  RandomHyperplanes.resize(NumPlanes);
  for (size_t i = 0; i < NumPlanes; ++i) {
    torch::Tensor hyperplane = torch::normal(0, 1, {static_cast<long>(Dimensions)});
    RandomHyperplanes[i] = hyperplane / hyperplane.norm();
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

std::vector<std::string> LSHSearch::BatchHashFunction(const torch::Tensor& batch) {
  std::vector<std::string> hashValues;
  hashValues.reserve(batch.size(0));

  torch::Tensor batch_dot_products = torch::matmul(batch, torch::stack(RandomHyperplanes));

  for (int64_t i = 0; i < batch.size(0); ++i) {
    std::string hashValue = "";
    auto row_dot_products = batch_dot_products[i];

    for (int j = 0; j < RandomHyperplanes.size(); ++j) {
      hashValue += (row_dot_products[j].item<float>() > 0.001) ? '1' : '0';
    }
    hashValues.push_back(hashValue);
  }

  return hashValues;
}

bool LSHSearch::loadInitialTensor(torch::Tensor& t) {
  if (idToBucket.empty()) {
    idToBucket.reserve(t.size(0));
  }

  std::vector<std::string> hashValues = BatchHashFunction(t);

  for (int64_t i = 0; i < t.size(0); ++i) {
    auto row = t[i];
    int64_t id = ++lastNNZ;
    std::string hashValue = hashValues[i];

    Index[hashValue][id] = row;
    idToBucket[id] = hashValue;
  }

  return true;
}

}  // namespace CANDY_ALGO
