//
// Created by zhonghao on 27/11/24.
//
#include "Algorithms/Manu/Execution/QueryNode.hpp"
#include "Algorithms/Manu/Storage/DataStore.hpp"

QueryNode::QueryNode(Datastore* ds) : datastore(ds), lastTimetick() {}

void QueryNode::handleQuery(const WAL& wal) {
  if (wal.walType == WAL::Type::SEARCH) {
    // TODO: Implement consistency checks using timeticks and perform AKNN search
  }
}
