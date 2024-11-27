//
// Created by zhonghao on 27/11/24.
//

#ifndef MANU_BINLOG_HPP
#define MANU_BINLOG_HPP

#include <string>
#include <vector>

class Binlog {
public:
  std::string label;
  std::vector<std::string> vectorIDs;

  Binlog(const std::string& label, const std::vector<std::string>& vectorIDs);
};

#endif // MANU_BINLOG_HPP
