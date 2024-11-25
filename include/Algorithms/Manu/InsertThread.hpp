//
// Created by zhonghao on 25/11/24.
//

#ifndef INSERT_THREAD_HPP
#define INSERT_THREAD_HPP

#include "ExecutorThread.hpp"
#include "WriteAheadLog.hpp"
#include "IndexManager.hpp"
#include "BinLog.hpp"
#include "DataStore.hpp"

class InsertThread : public ExecutorThread {
private:
  WriteAheadLog* wal;
  IndexManager* indexManager;
  BinLog* binLog;
  DataStore* dataStore;
  uint64_t startRow, endRow;

public:
  InsertThread(WriteAheadLog* wal, IndexManager* indexManager, BinLog* binLog, DataStore* dataStore, uint64_t startRow, uint64_t endRow);
  ~InsertThread();

protected:
  void execute() override;
};

#endif
