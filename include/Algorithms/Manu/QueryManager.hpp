//
// Created by zhonghao on 22/11/24.
// Process vector queries
//

#ifndef QUERY_MANAGER_HPP
#define QUERY_MANAGER_HPP

#include <string>
#include <vector>
#include "IndexManager.hpp"
#include "StateManager.hpp"

class QueryManager {
private:
  IndexManager* indexManager;
  StateManager* stateManager;

public:
  QueryManager(IndexManager* indexManager, StateManager* stateManager);

  std::vector<std::string> executeQuery(const std::string& queryParams);
  void coordinateSources();
};

#endif
