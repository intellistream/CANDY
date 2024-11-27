//
// Created by zhonghao on 27/11/24.
//
#include "Algorithms/Manu/Execution/DataNode.hpp"
#include "Algorithms/Manu/Storage/DataStore.hpp"

DataNode::DataNode(DataCoordinator* dc, Datastore* ds) : dataCoordinator(dc), datastore(ds) {}

void DataNode::handleWAL(const WAL& wal) {
  if (wal.walType == WAL::Type::INSERTION) {
    // TODO: Insert vector into a growing segment and manage segment threshold
  }
}
