//
// Created by zhonghao on 25/11/24.
//

#ifndef SEARCH_THREAD_HPP
#define SEARCH_THREAD_HPP

#include "ExecutorThread.hpp"
#include "QueryManager.hpp"

class SearchThread : public ExecutorThread {
private:
  QueryManager* queryManager;
  std::string queryParams;

public:
  SearchThread(QueryManager* queryManager, const std::string& queryParams);
  ~SearchThread();

protected:
  void execute() override;
};

#endif
