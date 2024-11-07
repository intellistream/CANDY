/*
 * Copyright (C) 2024 by the INTELLI team
 * Created on: 24-11-5 下午2:49
 * Description: ${DESCRIPTION}
 */

#include <torch/torch.h>
#include <Algorithms  //LSH/LSHSearch.hpp>
#include <algorithm>
#include <iostream>
#include <random>
#include <unordered_map>

namespace CANDY_ALGO {

// Constructor with vector dimensions and number of planes
LshSearch::LshSearch(size_t Dimensions, size_t NumPlanes)
    : Dimensions(Dimensions) {
  // Initialize the hyperplane
  GenerateRandomHyperplanes(NumPlanes);
}

// Set configuration
bool LshSearch::setConfig(INTELLI::ConfigMapPtr cfg) {
  ANNSBase::setConfig(cfg);
  return true;
}

// Reset the LSH data structure
void LshSearch::reset() {
  Index.clear();
  nearbyBuckets.clear();
  GlobalIndexCounter = 0;
}

// Insert tensor into LSH
bool LshSearch::insertTensor(const torch::Tensor& t) {
  for (int64_t i = 0; i < t.size(0); ++i) {
    auto row = t[i];
    int64_t id = GlobalIndexCounter++;
    std::string hashValue = HashFunction(row);
    Index[hashValue][id] = row;
  }
  return true;
}

// Delete tensor from LSH
bool LshSearch::deleteTensor(torch::Tensor& t, int64_t k) {
  auto results = searchTensor(t, k);

  for (int64_t i = 0; i < t.size(0); ++i) {
    std::vector<int64_t> indicesToDelete;
    for (int j = 0; j < k; ++j) {
      indicesToDelete.push_back(results[i][j].item<int64_t>());
    }

    auto& rowNearbyBuckets = nearbyBuckets[i];
    int deletedCount = 0;

    //std::cout << "搜索后第" << i <<"个张量的相邻桶的编号（前10个）：";
    //for (int l = 0; l < 10; ++l) {
    //  std::cout << rowNearbyBuckets[l] << " ";
    //}
    std::cout << std::endl;

    for (const auto& [dist, bucket] : rowNearbyBuckets) {

      //std::cout << "**（正在删除）-当前所在桶号：" << bucket << endl;
      //std::cout << "***当前已经删除个数：" << deletedCount <<endl;

      if (deletedCount >= k) {
        break;
      }

      if (Index.count(bucket) > 0) {
        auto& BucketMap = Index[bucket];

        for (int64_t index = deletedCount; index < indicesToDelete.size();
             ++index) {
          auto id = indicesToDelete[index];

          if (BucketMap.count(id) > 0) {
            BucketMap.erase(id);
            deletedCount++;

            if (deletedCount >= k) {
              break;
            }
          } else {
            break;
          }
        }
      }
    }
  }
  return true;
}

// Revise tensor (modify its value)
bool LshSearch::reviseTensor(torch::Tensor& t, torch::Tensor& w) {
  if (t.size(0) > w.size(0) || t.size(1) != w.size(1)) {
    return false;
  }

  for (int64_t i = 0; i < t.size(0); ++i) {
    auto Row = t[i];
    std::string Bucket = HashFunction(Row);

    if (Index.count(Bucket) > 0) {
      auto& BucketMap = Index[Bucket];

      for (auto [TensorId, Tensor] : BucketMap) {
        if (torch::equal(Tensor, Row)) {
          BucketMap.erase(TensorId);
          std::string NewBucket = HashFunction(w[i]);

          if (NewBucket != Bucket) {
            Index[NewBucket][TensorId] = w[i];
          } else {
            BucketMap[TensorId] = w[i];
          }
          break;
        }
      }
    }
  }
  return true;
}

// Search for the k nearest neighbors of tensor q
std::vector<torch::Tensor> LshSearch::searchTensor(const torch::Tensor& q,
                                                   int64_t k) {
  std::vector<torch::Tensor> Results;
  nearbyBuckets.clear();  // Clear nearbyBuckets before every search

  for (int64_t i = 0; i < q.size(0); ++i) {
    auto Row = q[i];
    std::string Bucket = HashFunction(Row);

    std::vector<size_t> Indices;
    std::vector<std::pair<float, size_t>> Distances;

    // Use Hamming distance to get nearby buckets
    std::vector<std::pair<float, std::string>> rowNearbyBuckets;

    for (const auto& [otherBucket, _] : Index) {
      int dist = HammingDistance(Bucket, otherBucket);
      rowNearbyBuckets.push_back({static_cast<float>(dist), otherBucket});
    }

    std::sort(rowNearbyBuckets.begin(), rowNearbyBuckets.end());
    nearbyBuckets.push_back(rowNearbyBuckets);

    //std::cout << "搜索后第" << i <<"个张量的相邻桶的编号（前10个）：";
    //for (int l = 0; l < 10; ++l) {
    //  std::cout << rowNearbyBuckets[l] << " " ;
    //}
    //std::cout << std::endl;

    // Traverse all the buckets and compute distances, until we find k closest tensors
    for (const auto& [dist, nBucket] : rowNearbyBuckets) {

      //std::cout << "当前查询所在的桶号： " << nBucket << std::endl;

      if (Indices.size() >= k) {
        break;
      }

      if (Index.count(nBucket) > 0) {
        auto& BucketMap = Index[nBucket];
        for (const auto& [TensorId, Tensor] : BucketMap) {
          float Distance = (Tensor - Row).norm().item<float>();
          Distances.emplace_back(Distance, TensorId);
        }

        // Sort the distances
        std::sort(Distances.begin(), Distances.end());

        // Add all elements from Distances to Indices
        for (int j = 0; j < static_cast<int64_t>(Distances.size()); ++j) {
          Indices.push_back(Distances[j].second);
        }

        // Clear the Distances for the next round
        Distances.clear();
      }
    }

    // If Indices size exceeds k, truncate it to the first k elements
    if (Indices.size() > k) {
      Indices.resize(k);
    }

    // Convert indices to a tensor and add to results
    torch::Tensor Tensor = torch::empty({static_cast<long>(Indices.size()), 1},
                                        torch::dtype(torch::kLong));
    for (size_t j = 0; j < Indices.size(); ++j) {
      Tensor[j][0] = static_cast<long>(Indices[j]);
    }
    Results.push_back(Tensor);
  }

  return Results;
}

// Generate random hyperplanes for hashing
void LshSearch::GenerateRandomHyperplanes(size_t NumPlanes) {
  RandomHyperplanes.resize(NumPlanes);
  for (size_t i = 0; i < NumPlanes; ++i) {
    RandomHyperplanes[i] =
        torch::empty({static_cast<long>(Dimensions)}).uniform_(-1, 1);
  }
}

// Hash function to map a tensor to a bucket (returns string hash)
std::string LshSearch::HashFunction(const torch::Tensor& t) {
  std::string hashValue = "";

  for (const auto& plane : RandomHyperplanes) {
    float dotProduct = torch::dot(t, plane).item<float>();
    hashValue += (dotProduct > 0) ? '1' : '0';
  }

  return hashValue;
}

// Hamming distance between two binary strings
int LshSearch::HammingDistance(const std::string& str1,
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
