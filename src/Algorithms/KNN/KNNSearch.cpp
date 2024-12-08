/*
 * Copyright (C) 2024 by the INTELLI team
 * Created on: 2024/10/9
 * Description: [Provide description here]
 */
#include <Algorithms/KNN/KNNSearch.hpp>
#include <Utils/Computation.hpp>
#include <Utils/TensorOP.hpp>

// Constructor with vector dimensions

CANDY_ALGO::KnnSearch::KnnSearch(size_t dimensions) : dimensions(dimensions) {}

bool CANDY_ALGO::KnnSearch::setConfig(INTELLI::ConfigMapPtr cfg) {
  //ANNSBase::setConfig(cfg);
  if (cfg == nullptr) {
    dbTensor = torch::zeros({initialVolume, vecDim});
    lastNNZ = -1;
    return false;
  }
  vecDim = cfg->tryI64("vecDim", 768, true);
  initialVolume = cfg->tryI64("initialVolume", 1000, true);
  expandStep = cfg->tryI64("expandStep", 100, true);
  dbTensor = torch::zeros({initialVolume, vecDim});
  lastNNZ = -1;
  return true;
}

// Reset the current index

void CANDY_ALGO::KnnSearch::reset() {

  index.clear();
}

// Insert tensor into the index

bool CANDY_ALGO::KnnSearch::insertTensor(const torch::Tensor& t) {

  return INTELLI::TensorOP::appendRowsBufferMode(&dbTensor, &t, &lastNNZ,
                                                 expandStep);
}

// Delete tensor from the index

bool CANDY_ALGO::KnnSearch::deleteTensor(torch::Tensor& t, int64_t k) {

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

  // Delete rows using INTELLI::TensorOP::deleteRowsBufferMode
  return INTELLI::TensorOP::deleteRowsBufferMode(&dbTensor, idxToDelete,
                                                 &lastNNZ);
}

bool CANDY_ALGO::KnnSearch::reviseTensor(torch::Tensor& t, torch::Tensor& w) {

  // Check if dimensions match
  if (t.size(0) > w.size(0) || t.size(1) != w.size(1)) {
    return false;
  }

  // Ensure dbTensor and t are contiguous for compatibility with cdist
  torch::Tensor dbData = dbTensor.contiguous();
  torch::Tensor queryData = t.contiguous();

  // Compute pairwise distances between queryData (t) and dbData (dbTensor)
  torch::Tensor distances =
      torch::cdist(queryData, dbData);  // Shape: (rows, dbSize)

  // Iterate over each row in t to find and revise the nearest neighbor in dbTensor
  for (int64_t i = 0; i < t.size(0); ++i) {
    // Find the index of the nearest neighbor for the current row in t
    int64_t nearestIdx = std::get<1>(distances[i].min(0)).item<int64_t>();

    // Ensure the index is within the bounds of lastNNZ
    if (0 <= nearestIdx && nearestIdx <= lastNNZ) {
      // Slice the corresponding row from w
      auto rowW = w.slice(0, i, i + 1);

      // Update dbTensor at the nearest neighbor index with the new data from w
      INTELLI::TensorOP::editRows(&dbTensor, &rowW, nearestIdx);
    }
  }

  return true;
}

std::vector<torch::Tensor> CANDY_ALGO::KnnSearch::searchTensor(
    const torch::Tensor& q, int64_t k) {

  // Ensure dbTensor is contiguous in memory
  torch::Tensor dbData = dbTensor.contiguous();
  torch::Tensor queryData = q.contiguous();

  // 获取dbData 的有效数据
  long index = lastNNZ + 1;
  if (index < dbData.size(0)) {
    dbData = dbData.slice(0, 0, index);
  }

  // Check if the database contains any entries
  if (dbData.size(0) == 0) {
    // Return an empty vector if the database is empty
    std::vector<torch::Tensor> results(queryData.size(0),
                                       torch::empty({0}, torch::kInt64));
    return results;
  }

  // Compute pairwise distances between the query tensor and dbTensor
  torch::Tensor distances =
      torch::cdist(queryData, dbData);  // Shape: (rows, dbSize)

  // Prepare vector to hold results
  std::vector<torch::Tensor> results;

  // For each query, retrieve the top-k nearest neighbors
  for (int64_t i = 0; i < queryData.size(0); ++i) {
    // Clamp k to avoid exceeding the available number of neighbors
    int64_t clamped_k = std::min(k, distances.size(1));

    // Find the indices of the top-k smallest distances for the current query row
    auto topk =
        std::get<1>(distances[i].topk(clamped_k, -1, /*largest=*/false));
    // Store the indices of top-k nearest neighbors for this query row
    results.push_back(topk);
  }

  return results;
}

