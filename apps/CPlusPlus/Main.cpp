/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/9
 * Description: Implementation for streaming insert of tensors using an approximate nearest neighbor search (ANNS) system.
 */

#include <Performance/monitoring.hpp> // Performance utilities
#include <iostream>
#include <chrono>
#include <string>
#include <Core/vector_db.hpp>
#include <Utils/UtilityFunctions.h>
#include <DataLoader/DataLoaderTable.hpp>
#include <Algorithms/KNN/KNNSearch.hpp>

using namespace INTELLI;
using namespace std;
static inline std::vector<INTELLI::IntelliTimeStampPtr> timeStamps;
static inline timer_t timerid;

bool fileExists(const std::string &filename) {
  std::ifstream file(filename);
  return file.good(); // Returns true if the file is open and in a good state
}

static inline int64_t s_timeOutSeconds = -1;

int main(int argc, char **argv) {
  /**
   * @brief 1. Load the configs
   */
  INTELLI::ConfigMapPtr inMap = newConfigMap();
  if (inMap->fromCArg(argc, argv) == false) {
    if (argc >= 2) {
      std::string fileName = "";
      fileName += argv[1];
      if (inMap->fromFile(fileName)) {
        std::cout << "Config loaded from file " + fileName << endl;
      }
    }
  }

  /**
   * @brief 2. Create the data and query, and prepare initialTensor
   */
  CANDY::DataLoaderTable dataLoaderTable;
  std::string dataLoaderTag = inMap->tryString("dataLoaderTag", "random", true);
  int64_t cutOffTimeSeconds = inMap->tryI64("cutOffTimeSeconds", -1, true);
  int64_t waitPendingWrite = inMap->tryI64("waitPendingWrite", 0, true);
  auto dataLoader = dataLoaderTable.findDataLoader(dataLoaderTag);
  INTELLI_INFO("2. Data loader = " + dataLoaderTag);
  if (dataLoader == nullptr) {
    INTELLI_ERROR("Data loader not found: " + dataLoaderTag);
    return -1;
  }
  dataLoader->setConfig(inMap);
  int64_t initialRows = inMap->tryI64("initialRows", 0, true);
  auto dataTensorAll = dataLoader->getData().nan_to_num(0);
  auto dataTensorInitial = dataTensorAll.slice(0, 0, initialRows);
  auto dataTensorStream = dataTensorAll.slice(0, initialRows, dataTensorAll.size(0));
  auto queryTensor = dataLoader->getQuery().nan_to_num(0);

  INTELLI_INFO(
    "Initial tensor: Dimension = " + std::to_string(dataTensorInitial.size(1)) + ", #data = "
    + std::to_string(dataTensorInitial.size(0)));
  INTELLI_INFO(
    "Streaming tensor: Dimension = " + std::to_string(dataTensorStream.size(1)) + ", #data = "
    + std::to_string(dataTensorStream.size(0)) + ", #query = "
    + std::to_string(queryTensor.size(0)));

  /**
   * @brief 3. Create the timestamps
   */
  INTELLI::IntelliTimeStampGenerator timeStampGen;
  inMap->edit("streamingTupleCnt", (int64_t) dataTensorStream.size(0));
  timeStampGen.setConfig(inMap);
  timeStamps = timeStampGen.getTimeStamps();
  INTELLI_INFO("3. TimeStamp Size = " + std::to_string(timeStamps.size()));
  int64_t batchSize = inMap->tryI64("batchSize", dataTensorStream.size(0), true);

  /**
   * @brief 4. Create index (ANNS Index Initialization)
   */
  size_t dimensions = dataTensorInitial.size(1);
  ANNSBasePtr indexPtr = std::make_shared<KnnSearch>(dimensions);
  if (!indexPtr->setConfig(inMap)) {
    INTELLI_ERROR("Failed to configure ANNS index.");
    return -1;
  }

  /**
   * @brief 5. Streaming feed
   */
  uint64_t startRow = 0;
  uint64_t endRow = startRow + batchSize;
  uint64_t tNow = 0;
  uint64_t tExpectedArrival = timeStamps[endRow - 1]->arrivalTime;
  uint64_t tp = 0;
  uint64_t tDone = 0;
  uint64_t aRows = dataTensorStream.size(0);
  INTELLI_INFO("4.0 Load initial tensor!");
  if (initialRows > 0) {
    indexPtr->loadInitialTensor(dataTensorInitial);
  }
  auto start = std::chrono::high_resolution_clock::now();

  INTELLI_INFO("4.1 STREAMING NOW!!!");
  double processedOld = 0;
  while (startRow < aRows) {
    tNow = chronoElapsedTime(start);
    while (tNow < tExpectedArrival) {
      tNow = chronoElapsedTime(start);
    }
    double processed = endRow;
    processed = processed * 100.0 / aRows;

    /**
     * @brief Now, the whole batch has arrived, compute
     */
    auto subA = dataTensorStream.slice(0, startRow, endRow);
    indexPtr->insertTensor(subA);
    tp = chronoElapsedTime(start);

    /**
     * @brief The new arrived A will be no longer probed, so we can assign the processed time now
     */
    for (size_t i = startRow; i < endRow; i++) {
      timeStamps[i]->processedTime = tp;
    }

    /**
     * @brief Update the indexes
     */
    startRow += batchSize;
    endRow += batchSize;
    if (endRow >= aRows) {
      endRow = aRows;
    }
    if (processed - processedOld >= 10.0) {
      INTELLI_INFO("Done " + to_string(processed) + "% (" + to_string(startRow) + "/" + to_string(aRows) + ")");
      processedOld = processed;
    }
    tExpectedArrival = timeStamps[endRow - 1]->arrivalTime;
  }
  tDone = chronoElapsedTime(start);

  /**
   * @brief Validation and summary
   */
  int64_t ANNK = inMap->tryI64("ANNK", 5, true);
  int64_t pendingWriteTime = 0;

  INTELLI_INFO("Insert is done, let us validate the results");

  double throughput = aRows * 1e6 / tDone;
  double throughputByElements = throughput * dataTensorStream.size(1);
  double latency95 = UtilityFunctions::getLatencyPercentage(0.95, timeStamps);
  auto briefOutCfg = newConfigMap();
  briefOutCfg->edit("throughput", throughput);
  briefOutCfg->edit("throughputByElements", throughputByElements);
  briefOutCfg->edit("95%latency(Insert)", latency95);
  briefOutCfg->edit("pendingWrite", pendingWriteTime);
  briefOutCfg->edit("normalExit", (int64_t) 1);
  briefOutCfg->toFile("onlineInsert_result.csv");
  std::cout << "Brief results\n" << briefOutCfg->toString() << std::endl;
  UtilityFunctions::saveTimeStampToFile("onlineInsert_timestamps.csv", timeStamps);

  return 0;
}
