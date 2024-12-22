/*
 * Copyright (C) 2024 by the jjzhao
 * Created on: 2024/12/22
 * Description: [Provide description here]
 */
#include <Algorithms/KNN/SeparateKNNSearch.hpp>
#include <Utils/TensorOP.hpp>

namespace CANDY_ALGO {
SeparateKNNSearch::SeparateKNNSearch(size_t dimensions) : dimensions(dimensions) {}
bool SeparateKNNSearch::setConfig(INTELLI::ConfigMapPtr cfg) {
  if (cfg == nullptr) {
    dbTensor = torch::zeros({initialVolume, vecDim});
    lastNNZ = -1;
    return false;
  }
  vecDim = cfg-> tryI64("vecDim", 768, true);
  initialVolume = cfg->tryI64("initialVolume", 1000, true);
  expandStep = cfg->tryI64("expandStep", 100, true);
  dbTensor = torch::zeros({initialVolume, vecDim});
  lastNNZ = -1;
  return true;
}
bool SeparateKNNSearch::insertTensor(const torch::Tensor& t) {
  return this -> storage_engine -> insertTensor(t);
}
std::vector<torch::Tensor> SeparateKNNSearch::searchTensor(const torch::Tensor& t, int64_t k) {
  vector<torch::Tensor> all_vector = this -> storage_engine -> getAll();
  for (torch::Tensor a : all_vector) {
    INTELLI::TensorOP::appendRowsBufferMode(&dbTensor, &a, &lastNNZ, expandStep);
  }
  return findKnnTensorBurst(t, k);
}
std::vector<torch::Tensor> SeparateKNNSearch::deleteTensor(const torch::Tensor& t, int64_t k) {
  // Use the searchTensor function to get the indices of k-nearest neighbors for each row in t
  std::vector<torch::Tensor> idxToDeleteTensors = searchTensor(t, k);

  // Flatten idxToDeleteTensors into a single vector of int64_t
  std::vector<int64_t> idxToDelete;
  for (const auto& tensor : idxToDeleteTensors) {
    auto tensorAccessor = tensor.accessor<int64_t, 1>();
    for (int64_t i = 0; i < tensor.size(0); ++i) {
      idxToDelete.push_back(tensorAccessor[i]);
    }
  }
  return this -> storage_engine -> deleteTensor(idxToDelete);
}
bool SeparateKNNSearch::reviseTensor(const torch::Tensor& t, const torch::Tensor& w) {
  // Check if dimensions match
  if (t.size(0) > w.size(0) || t.size(1) != w.size(1)) {
    return false;
  }
  // Use the searchTensor function to get the indices of k-nearest neighbors for each row in t
  std::vector<torch::Tensor> idxToDeleteTensors = searchTensor(t, 1);
  for (int64_t i = 0; i < t.size(0); ++i) {
    auto tensorAccessor = idxToDeleteTensors[i].accessor<int64_t, 1>();
    auto rowW = w.slice(0, i, i + 1);
    this -> storage_engine -> reviseTensor(rowW, tensorAccessor[0]);
  }
}
std::vector<torch::Tensor> SeparateKNNSearch::findKnnTensorBurst(const torch::Tensor& q, int64_t k) {
  // Ensure dbTensor is contiguous in memory
  torch::Tensor dbData = dbTensor.contiguous();
  torch::Tensor queryData = q.contiguous();

  // Retrieve the valid data from dbData
  long index = lastNNZ + 1;
  if (index < dbData.size(0)) {
    dbData = dbData.slice(0, 0, index);
  }

  // Check if the database contains any entries
  if (dbData.size(0) == 0) {
    // Return an empty vector if the database is empty
    return {};
  }

  // Compute pairwise distances between the query tensor and dbTensor
  torch::Tensor distances =
      torch::cdist(queryData, dbData);  // Shape: (query_size, db_size)

  // Prepare vector to hold results
  std::vector<torch::Tensor> results;

  // For each query, retrieve the top-k nearest neighbors
  for (int64_t i = 0; i < queryData.size(0); ++i) {
    // Clamp k to avoid exceeding the available number of neighbors
    int64_t clamped_k = std::min(k, distances.size(1));

    // Find the indices of the top-k smallest distances for the current query row
    auto topk =
        std::get<1>(distances[i].topk(clamped_k, -1, /*largest=*/false));

    // Retrieve the actual tensors for these indices
    for (int64_t j = 0; j < clamped_k; ++j) {
      results.push_back(dbData[topk[j]]);
    }
  }
  INTELLI_INFO(results.size());
  dbTensor = torch::zeros({initialVolume, vecDim});
  return results;
}
}  // namespace CANDY_ALGO