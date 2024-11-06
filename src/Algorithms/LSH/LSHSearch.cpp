/*
 * Copyright (C) 2024 by the INTELLI team
 * Created on: 24-11-5 下午2:49
 * Description: ${DESCRIPTION}
 */

#include <torch/torch.h>
#include <Algorithms/LSH/LSHSearch.hpp>
#include <algorithm>
#include <cmath>

CANDY_ALGO::LshSearch::LshSearch(size_t Dimensions, size_t NumPlanes)
    : Dimensions(Dimensions) {
  GenerateRandomHyperplanes(NumPlanes);
}

bool CANDY_ALGO::LshSearch::setConfig(INTELLI::ConfigMapPtr cfg) {
  ANNSBase::setConfig(cfg);
  return true;
}

void CANDY_ALGO::LshSearch::reset() {
  Index.clear();
}

bool CANDY_ALGO::LshSearch::insertTensor(const torch::Tensor& t) {
  if (t.size(1) != Dimensions) {
    return false;
  }
  for (int64_t i = 0; i < t.size(0); ++i) {
    size_t Bucket = HashFunction(t[i]);
    Index[Bucket][GlobalIndexCounter++] = t[i];
  }
  return true;
}

bool CANDY_ALGO::LshSearch::deleteTensor(torch::Tensor& t, int64_t k) {
  std::vector<torch::Tensor> NearestIndices = searchTensor(t, k);

  for (int64_t i = 0; i < NearestIndices.size(); ++i) {
    auto Indices = NearestIndices[i];
    size_t Bucket = HashFunction(t[i]);

    if (Index.count(Bucket) > 0) {
      auto& BucketMap = Index[Bucket];

      // For each found index, delete the corresponding tensor
      for (int j = 0; j < Indices.size(0); ++j) {
        size_t TensorId = Indices[j][0].item<int64_t>();
        BucketMap.erase(TensorId);
      }
    }
  }
  return true;
}

bool CANDY_ALGO::LshSearch::reviseTensor(torch::Tensor& t, torch::Tensor& w) {
  if (t.size(0) > w.size(0) || t.size(1) != w.size(1)) {
    return false;
  }

  for (int64_t i = 0; i < t.size(0); ++i) {
    auto Row = t[i];
    size_t Bucket = HashFunction(Row);

    if (Index.count(Bucket) > 0) {
      auto& BucketMap = Index[Bucket];

      for (auto [TensorId, Tensor] : BucketMap) {
        if (torch::equal(Tensor, Row)) {
          BucketMap.erase(TensorId);
          size_t NewBucket = HashFunction(w[i]);

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

std::vector<torch::Tensor> CANDY_ALGO::LshSearch::searchTensor(
    const torch::Tensor& q, int64_t k) {
  std::vector<torch::Tensor> Results;
  // std::cout << "q.size(0): " << q.size(0) << ", q.size(1): " << q.size(1) << std::endl;
  for (int64_t i = 0; i < q.size(0); ++i) {
    auto Row = q[i];
    size_t Bucket = HashFunction(Row);

    std::vector<size_t> Indices;

    if (Index.count(Bucket) > 0) {
      auto& BucketMap = Index[Bucket];
      std::vector<std::pair<float, size_t>> Distances;

      for (const auto& [TensorId, Tensor] : BucketMap) {
        float Distance = (Tensor - Row).norm().item<float>();
        Distances.emplace_back(Distance, TensorId);
      }

      // Sort and store k with the smallest distance
      std::sort(Distances.begin(), Distances.end());

      for (int j = 0; j < std::min(k, static_cast<int64_t>(Distances.size()));
           ++j) {
        Indices.push_back(Distances[j].second);
      }
    }

    // Modify the results to ensure the returned tensor has shape (K, 1)
    if (!Indices.empty()) {
      torch::Tensor Tensor = torch::empty(
          {static_cast<long>(Indices.size()), 1}, torch::dtype(torch::kLong));

      for (size_t j = 0; j < Indices.size(); ++j) {
        Tensor[j][0] = static_cast<long>(Indices[j]);
      }

      Results.push_back(Tensor);
    } else {
      Results.push_back(torch::empty({0, 1}, torch::dtype(torch::kLong)));
    }
  }

  return Results;
}

void CANDY_ALGO::LshSearch::GenerateRandomHyperplanes(size_t numPlanes) {
  // std::mt19937 gen(std::random_device{}());
  // std::normal_distribution<float> dist(0.0f, 1.0f);

  RandomHyperplanes.resize(numPlanes);
  for (size_t i = 0; i < numPlanes; ++i) {
    RandomHyperplanes[i] =
        torch::empty({static_cast<long>(Dimensions)}).uniform_(-1, 1);
  }
}

size_t CANDY_ALGO::LshSearch::HashFunction(const torch::Tensor& t) {
  size_t HashValue = 0;

  for (const auto& hyperplane : RandomHyperplanes) {
    float dotProduct = (t * hyperplane).sum().item<float>();
    HashValue = (HashValue << 1) | (dotProduct > 0 ? 1 : 0);
  }

  return HashValue;
}
