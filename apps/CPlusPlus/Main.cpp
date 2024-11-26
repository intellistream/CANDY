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
#include "Algorithms/HNSW/hnsw.hpp"
using namespace INTELLI;
using namespace std;

const std::string candy_path = CANDY_PATH;

int main(int argc, char** argv) {
  /**
    * @brief 1. Load the configuration
    */
  ConfigMapPtr inMap = newConfigMap();
  std::string fileName =
      (argc >= 2) ? argv[1] : candy_path + "/config/configHNSW.csv";
  if (inMap->fromFile(fileName)) {
    INTELLI_INFO("Config loaded from file: " + fileName);
  } else {
    INTELLI_ERROR("Failed to load config from file: " + fileName);
    return -1;
  }

  /**
    * @brief 2. Load data
    */

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

  /**
    * @brief 3. Create timestamps
    */
  TimeStampGenerator timeStampGen;
  inMap->edit("streamingTupleCnt",
              static_cast<int64_t>(dataTensorStream.size(0)));
  timeStampGen.setConfig(inMap);
  auto timeStamps = timeStampGen.getTimeStamps();
  INTELLI_INFO("Timestamps generated: Size = " +
               std::to_string(timeStamps.size()));

  /**
    * @brief 4. Create index (ANNS Index Initialization)
    */
  size_t dimensions = dataTensorStream.size(1);
  auto algorithmTable = std::make_shared<CANDY_ALGO::AlgorithmTable>();

  std::string indexTag = inMap->tryString("indexTag", "KNN", true);
  auto indexPtr = algorithmTable->getIndex(indexTag);
  if (!indexPtr->setConfig(inMap)) {
    INTELLI_ERROR("Failed to configure ANNS index.");
    return -1;
  }

  /**
    * @brief 5. Feed streaming data
    */
  INTELLI_INFO("Feeding streaming data...");
  auto start = std::chrono::high_resolution_clock::now();

  int64_t initialRows = inMap->tryI64("initialRows", 0, true);
  int64_t batchSize =
      inMap->tryI64("batchSize", dataTensorStream.size(0), true);

  auto dataTensorInitial = dataTensorAll.slice(0, 0, initialRows);

  uint64_t startRow = 0;
  uint64_t endRow = startRow + batchSize;
  uint64_t aRows = dataTensorStream.size(0);

  INTELLI_INFO("3.0 Load initial tensor!");
  if (initialRows > 0) {
    indexPtr->loadInitialTensor(dataTensorInitial);
  }

  INTELLI_INFO("3.1 STREAMING NOW!!!");

  double processedOld = 0;
  while (startRow < aRows) {
    /**
        * @brief The whole batch is ready, proceed with insertion
        */
    auto subBatch = dataTensorStream.slice(0, startRow, endRow);
    if (!indexPtr->insertTensor(subBatch)) {
      INTELLI_ERROR("Failed to insert batch starting at row: " +
                    std::to_string(startRow));
      return false;  // Optionally handle insertion failure
    }

    /**
        * @brief Update the indexes
        */
    startRow += batchSize;
    endRow += batchSize;
    if (endRow >= aRows) {
      endRow = aRows;
    }

    // Log progress for every 10% increment
    double processed = startRow * 100.0 / aRows;
    if (processed - processedOld >= 1.0) {
      INTELLI_INFO("Done " + std::to_string(processed) + "% (" +
                   std::to_string(startRow) + "/" + std::to_string(aRows) +
                   ")");
      processedOld = processed;
    }
  }
  /**
    * @brief 6. Wait until feed ends and display performance metrics
    */
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  INTELLI_INFO("Streaming feed completed in " + std::to_string(duration) +
               " ms.");

  INTELLI_INFO(
      "Streaming feed is done! Let us search and validate the results!");
  INTELLI_INFO("Insert is done, let us validate the results");
  int64_t ANNK = inMap->tryI64("ANNK", 5, true);
  auto startQuery = std::chrono::high_resolution_clock::now();
  auto indexResults = indexPtr->searchTensor(queryTensor, ANNK);
  uint64_t queryLatency = chronoElapsedTime(startQuery);
  INTELLI_INFO("Query done in " + to_string(queryLatency / 1000) + "ms");
  std::string groundTruthPrefix =
      inMap->tryString("groundTruthPrefix", "onlineInsert_GroundTruth", true);

  std::string probeName = groundTruthPrefix + "/" +
                          std::to_string(indexResults.size() - 1) + ".rbt";
  double recall = 0.0;

  int64_t groundTruthRedo = inMap->tryI64("groundTruthRedo", 1, true);

  if (std::ifstream(probeName).good() && (groundTruthRedo == 0)) {
    INTELLI_INFO("Ground truth exists, so I load it");
    auto gdResults = UtilityFunctions::tensorListFromFile(groundTruthPrefix,
                                                          indexResults.size());
    INTELLI_INFO("Ground truth is loaded");
    recall = UtilityFunctions::calculateRecall(gdResults, indexResults);
  } else {
    INTELLI_INFO("Ground truth does not exist, so I'll create it");
    auto gdMap = newConfigMap();
    gdMap->loadFrom(*inMap);
    auto gdIndex = std::make_shared<CANDY_ALGO::KnnSearch>(dimensions);
    gdIndex->setConfig(gdMap);
    if (initialRows > 0) {
      gdIndex->loadInitialTensor(dataTensorInitial);
    }
    gdIndex->insertTensor(dataTensorStream);

    auto gdResults = gdIndex->searchTensor(queryTensor, ANNK);
    INTELLI_INFO("Ground truth is done");

    recall = UtilityFunctions::calculateRecall(gdResults, indexResults);
    //UtilityFunctions::tensorListToFile(gdResults, groundTruthPrefix);
  }

  INTELLI_INFO("RECALL = " + std::to_string(recall));
  INTELLI_INFO("Query Latency = " + std::to_string(queryLatency));

  return 0;
}
