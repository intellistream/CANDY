/*
* Copyright (C) 2024 by the INTELLI team
* Created on: 2024/10/9
* Description: Simplified main program for streaming insert of tensors, as an introductory entry point for new developers.
*/

#include <Algorithms/AlgorithmTable.hpp>
#include <Algorithms/FlatGPUIndex/FlatGPUIndex.hpp>
#include <Algorithms/KNN/KNNSearch.hpp>
#include <Core/vector_db.hpp>
#include <DataLoader/DataLoaderTable.hpp>
#include <Utils/ConfigMap.hpp>
#include <Utils/TimeStampGenerator.hpp>
#include <Utils/UtilityFunctions.hpp>
#include <chrono>
#include <iostream>
#include <string>
#include <atomic>
#include <mutex>
#include "Utils/ThreadPool.hpp"
#include "Algorithms/HNSW/hnsw.hpp"
#include <torch/torch.h>

using namespace INTELLI;
using namespace std;

const std::string candy_path = CANDY_PATH;

int main(int argc, char** argv) {
  // 1. Load the configuration
  ConfigMapPtr inMap = newConfigMap();
  std::string fileName =
      (argc >= 2) ? argv[1] : candy_path + "/config/config.csv";
  if (inMap->fromFile(fileName)) {
    INTELLI_INFO("Config loaded from file: " + fileName);
  } else {
    INTELLI_ERROR("Failed to load config from file: " + fileName);
    return -1;
  }

  // 2. Load data
  CANDY_ALGO::DataLoaderTable dataLoaderTable;

  std::string dataLoaderTag = inMap->tryString("dataLoaderTag", "random", true);
  auto dataLoader = dataLoaderTable.findDataLoader(dataLoaderTag);
  if (dataLoader == nullptr) {
    INTELLI_ERROR("Data loader not found: " + dataLoaderTag);
    return -1;
  }
  dataLoader->setConfig(inMap);
  auto dataTensorAll = dataLoader->getData().nan_to_num(0);
  auto dataTensorStream = dataTensorAll;
  auto queryTensor = dataLoader->getQuery().nan_to_num(0);

  INTELLI_INFO(
      "Data loaded: Dimension = " + std::to_string(dataTensorStream.size(1)) +
      ", #data = " + std::to_string(dataTensorStream.size(0)));

  // 3. Create timestamps
  TimeStampGenerator timeStampGen;
  inMap->edit("streamingTupleCnt",
              static_cast<int64_t>(dataTensorStream.size(0)));
  timeStampGen.setConfig(inMap);
  auto timeStamps = timeStampGen.getTimeStamps();
  INTELLI_INFO("Timestamps generated: Size = " +
               std::to_string(timeStamps.size()));

  // 4. Create index (ANNS Index Initialization)
  size_t dimensions = dataTensorStream.size(1);
  auto algorithmTable = std::make_shared<CANDY_ALGO::AlgorithmTable>();

  std::string indexTag = inMap->tryString("indexTag", "KNN", true);
  auto indexPtr = algorithmTable->getIndex(indexTag);
  if (!indexPtr->setConfig(inMap)) {
    INTELLI_ERROR("Failed to configure ANNS index.");
    return -1;
  }

  // 5. Set up the thread pool
  int writeThreadCount = inMap->tryI64("writeThreadCount", 2, true);
  int readThreadCount = inMap->tryI64("readThreadCount", 2, true);
  int readTaskInterval = inMap->tryI64("readTaskInterval", 1, true); 

  ThreadPool pool(writeThreadCount + readThreadCount);
  pool.init();

  // 6. Prepare ground truth index
  int64_t batchSize = inMap->tryI64("batchSize", dataTensorStream.size(0), true);
  int64_t groundTruthRedo = inMap->tryI64("groundTruthRedo", 1, true);
  std::string groundTruthPrefix =
    inMap->tryString("groundTruthPrefix", "multiRW_GroundTruth", true);
  int64_t ANNK = inMap->tryI64("ANNK", 5, true);

  std::string probeName = 
    groundTruthPrefix + "/" + std::to_string(indexResults.size() - 1) + ".rbt";

  if (std::ifstream(probeName).good() && (groundTruthRedo == 0)) {
    auto gdResults = 
      UtilityFunctions::tensorListFromFile(groundTruthPrefix, indexResults.size());
    INTELLI_INFO("Ground truth is loaded");
    recall = UtilityFunctions::calculateRecall(gdResults, indexResults);
  } else {
    INTELLI_INFO("Ground truth does not exist");
    auto gdMap = newConfigMap();
    gdMap->loadFrom（*inMap);
    auto gdIndex = std::make_shared<CANDY_ALGO::KnnSearch>(dimensions);
    gdIndex->setConfig(gdMap);
    if (initialRows > 0) {
      gdIndex->loadInitialTensor(dataTensorInitial);
    }  
  }

  // 7. Concurrently feed streaming data and read results
  std::atomic<uint64_t> processedBatches{0};
  std::vector<double> recallValues;
  std::mutex recallMutex;

  int64_t totalRows = dataTensorStream.size(0);

  // Function to calculate recall
  auto calculateRecall = [&](const torch::Tensor& queryTensor, int64_t ANNK) {
    double recall = 0.0;


    
    {
      std::lock_guard<std::mutex> lock(recallMutex);
      recallValues.push_back(recall);
    }
  };

  // Writing thread function
  auto writeTask = [&](uint64_t startRow, uint64_t endRow) {
    auto subBatch = dataTensorStream.slice(0, startRow, endRow);
    if (!indexPtr->insertTensor(subBatch)) {
      INTELLI_ERROR("Failed to insert batch starting at row: " +
                    std::to_string(startRow));
    }
    processed_batches.fetch_add(1);
    INTELLI_INFO("Processed batch starting at row: " + std::to_string(startRow));
  };

  // Reading thread function
  auto readTask = [&]() {
    auto startQuery = std::chrono::high_resolution_clock::now();
    auto indexResults = indexPtr->searchTensor(queryTensor, ANNK);
    uint64_t queryLatency = chronoElapsedTime(startQuery);
    INTELLI_INFO("Query done in " + to_string(queryLatency / 1000) + "ms");

    calculateRecall(queryTensor, ANNK);
    INTELLI_INFO("Recall calculated during concurrent operations.");
  };

  INTELLI_INFO("Starting concurrent read/write...");
  auto start = std::chrono::high_resolution_clock::now();

  for (uint64_t startRow = 0; startRow < totalRows; startRow += batchSize) {
    uint64_t endRow = std::min(startRow + batchSize, totalRows);

    // Submit write tasks
    int64_t writeBatchSize = batchSize / writeThreadCount;
    uint64_t writeStartRow = startRow;
    for (int i = 0; i < writeThreadCount; i++) {
      startRow += taskBatchSize;
      pool.submit(writeTask, startRow, endRow);
    }
    
    // Submit read tasks intermittently
    int64_t readBatchSize = (writeBatchSize * writeThreadCount) / readThreadCount;
      for (int i = 0; i < readThreadCount; i++) {
      startRow += taskBatchSize;
      pool.submit(readTask, startRow, endRow);
    }
  }

  pool.shutdown();

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  INTELLI_INFO("Concurrent read/write completed in " + std::to_string(duration) + " ms.");

  return 0;
}
