//
// Created by zhonghao on 27/11/24.
//

#ifndef MANU_DATANODE_HPP
#define MANU_DATANODE_HPP

#include "../Coordination//DataCoordinator.hpp"
#include "../Log/Binlog.hpp"
#include "../Log/WAL.hpp"
#include "../Storage/DataStore.hpp"

class DataNode {
private:
  DataCoordinator* dataCoordinator;
  Datastore* datastore;

public:
  explicit DataNode(DataCoordinator* dc, Datastore* ds);
  void handleWAL(const WAL& wal);
};

#endif // MANU_DATANODE_HPP

