//
// Created by zhonghao on 25/11/24.
//
#include "Algorithms/Manu/InsertThread.hpp"

#include <iostream>


InsertThread::InsertThread(WriteAheadLog* wal, IndexManager* indexManager, BinLog* binLog, DataStore* dataStore, uint64_t startRow, uint64_t endRow)
    : wal(wal), indexManager(indexManager), binLog(binLog), dataStore(dataStore), startRow(startRow), endRow(endRow) {}

void InsertThread::execute() {
  std::vector<std::string> batch;

  for (uint64_t row = startRow; row < endRow && isRunning.load(); ++row) {
    std::string vectorData = "VectorData_" + std::to_string(row);

    wal->appendEntry(vectorData); // Write to WAL.
    dataStore->addVector(vectorData); // Add to datastore.

    batch.push_back(vectorData);

    if (batch.size() >= 10) {
      binLog->appendBatch(batch); // Write batch to BinLog.
      batch.clear();
      indexManager->updateIndex(dataStore->getAllVectors()); // Rebuild index.
    }
  }

  if (!batch.empty()) {
    binLog->appendBatch(batch); // Flush remaining batch.
    indexManager->updateIndex(dataStore->getAllVectors());
  }
}
