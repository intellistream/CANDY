//
// Created by zhonghao on 25/11/24.
//
#include "Algorithms/Manu/SearchThread.hpp"

#include <iostream>

SearchThread::SearchThread(QueryManager* queryManager, const std::string& queryParams)
    : queryManager(queryManager), queryParams(queryParams) {}

SearchThread::~SearchThread() {
  stop();
}

void SearchThread::execute() {
  while (isRunning.load()) {
    auto results = queryManager->executeQuery(queryParams);
    std::cout << "Search results: " << results.size() << " vectors found." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}
