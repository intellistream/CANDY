//
// Created by zhonghao on 27/11/24.
//

#ifndef MANU_QUERYNODE_HPP
#define MANU_QUERYNODE_HPP

#include "../Log/HybridClock.hpp"
#include "../Log/WAL.hpp"
#include "Algorithms/Manu/Storage/DataStore.hpp"

class QueryNode {
private:
  Datastore* datastore;
  HybridClock lastTimetick;

public:
  explicit QueryNode(Datastore* ds);
  void handleQuery(const WAL& wal);
};

#endif // MANU_QUERYNODE_HPP
