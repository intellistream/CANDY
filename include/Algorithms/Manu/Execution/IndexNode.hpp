//
// Created by zhonghao on 27/11/24.
//

#ifndef MANU_INDEXNODE_HPP
#define MANU_INDEXNODE_HPP

#include "../Log/Binlog.hpp"

class IndexNode {
public:
  void buildIndex(const Binlog& binlog);
};

#endif // MANU_INDEXNODE_HPP

