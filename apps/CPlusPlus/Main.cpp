/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/9
 * Description: Simplified main program for streaming insert of tensors, as an introductory entry point for new developers.
 */

#include <iostream>
#include <chrono>
#include <string>
#include <Core/vector_db.hpp>
#include <DataLoader/DataLoaderTable.hpp>

#include "Algorithms/KNN/KNNSearch.hpp"
#include "Utils/TimeStampGenerator.hpp"

using namespace INTELLI;
using namespace std;

int main(int argc, char **argv) {
  /**
   * @brief 1. Load the configuration
   */
  ConfigMapPtr inMap = newConfigMap();
  if (argc >= 2) {
    std::string fileName = argv[1];
    if (inMap->fromFile(fileName)) {
      INTELLI_INFO("Config loaded from file: " + fileName);
    } else {
      INTELLI_ERROR("Failed to load config from file: " + fileName);
      return -1;
    }
  } else {
    INTELLI_ERROR("No configuration file provided.");
    return -1;
  }

  /**
   * @brief 2. Load data
   */
  CANDY::DataLoaderTable dataLoaderTable;
  std::string dataLoaderTag = inMap->tryString("dataLoaderTag", "random", true);
  auto dataLoader = dataLoaderTable.findDataLoader(dataLoaderTag);
  if (dataLoader == nullptr) {
    INTELLI_ERROR("Data loader not found: " + dataLoaderTag);
    return -1;
  }
  dataLoader->setConfig(inMap);
  auto dataTensorAll = dataLoader->getData().nan_to_num(0);
  auto dataTensorStream = dataTensorAll;

  INTELLI_INFO("Data loaded: Dimension = " + std::to_string(dataTensorStream.size(1)) + ", #data = " + std::to_string(dataTensorStream.size(0)));

  /**
   * @brief 3. Create timestamps
   */
  TimeStampGenerator timeStampGen;
  inMap->edit("streamingTupleCnt", static_cast<int64_t>(dataTensorStream.size(0)));
  timeStampGen.setConfig(inMap);
  auto timeStamps = timeStampGen.getTimeStamps();
  INTELLI_INFO("Timestamps generated: Size = " + std::to_string(timeStamps.size()));

  /**
   * @brief 4. Create index (ANNS Index Initialization)
   */
  size_t dimensions = dataTensorStream.size(1);
  auto indexPtr = std::make_shared<KnnSearch>(dimensions);
  if (!indexPtr->setConfig(inMap)) {
    INTELLI_ERROR("Failed to configure ANNS index.");
    return -1;
  }

  /**
   * @brief 5. Feed streaming data
   */
  INTELLI_INFO("Feeding streaming data...");
  auto start = std::chrono::high_resolution_clock::now();

  // Insert tensors in a streaming manner
  for (int64_t i = 0; i < dataTensorStream.size(0); ++i) {
    auto singleTensor = dataTensorStream[i];
    if (!indexPtr->insertTensor(singleTensor)) {
      INTELLI_ERROR("Failed to insert tensor at index: " + std::to_string(i));
    }
  }

  /**
   * @brief 6. Wait until feed ends and display performance metrics
   */
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  INTELLI_INFO("Streaming feed completed in " + std::to_string(duration) + " ms.");

  return 0;
}
