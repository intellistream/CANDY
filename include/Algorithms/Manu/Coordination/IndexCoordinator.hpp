//
// Created by zhonghao on 27/11/24.
//

#ifndef MANU_INDEXCOORDINATOR_HPP
#define MANU_INDEXCOORDINATOR_HPP

#include "../Log/Binlog.hpp"

class IndexCoordinator {
public:
  void assignBinlogToIndexNode(const Binlog& binlog);
};

#endif // MANU_INDEXCOORDINATOR_HPP
