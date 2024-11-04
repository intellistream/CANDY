/*
 * Copyright (C) 2024 by the INTELLI team
 * Created on: 2024/10/9
 * Description: Simplified main program for streaming insert of tensors, as an introductory entry point for new developers.
 */

#include <iostream>
#include <chrono>
#include <string>
#include <Core/vector_db.hpp>
#include <DataLoader/DataLoaderTable.hpp>
#include <Algorithms/KNN/KNNSearch.hpp>
#include <Utils/TimeStampGenerator.hpp>

using namespace INTELLI;
using namespace std;

int main(int argc, char **argv) {
    /**
     * @brief 1. Load the configuration
     */
    ConfigMapPtr inMap = newConfigMap();
    std::string fileName = (argc >= 2) ? argv[1] : "config/config.csv";
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

    INTELLI_INFO(
        "Data loaded: Dimension = " + std::to_string(dataTensorStream.size(1)) + ", #data = " + std::to_string(
            dataTensorStream.size(0)));

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

    auto indexPtr = std::make_shared<CANDY_ALGO::KnnSearch>(dimensions);

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
    int64_t batchSize = inMap->tryI64("batchSize", dataTensorStream.size(0), true);

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
            INTELLI_ERROR("Failed to insert batch starting at row: " + std::to_string(startRow));
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
        double processed = endRow * 100.0 / aRows;
        if (processed - processedOld >= 10.0) {
            INTELLI_INFO("Done " + std::to_string(processed) + "% (" + std::to_string(startRow) + "/" + std::to_string(aRows) + ")");
            processedOld = processed;
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
