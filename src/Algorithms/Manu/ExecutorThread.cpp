//
// Created by zhonghao on 25/11/24.
//
#include "Algorithms/Manu/ExecutorThread.hpp"

ExecutorThread::ExecutorThread() : isRunning(false) {}

ExecutorThread::~ExecutorThread() {
  if (workerThread.joinable()) {
    stop();
    workerThread.join();
  }
}

void ExecutorThread::start() {
  isRunning = true;
  workerThread = std::thread(&ExecutorThread::execute, this);
}

void ExecutorThread::stop() {
  isRunning = false;
  if (workerThread.joinable()) {
    workerThread.join();
  }
}

bool ExecutorThread::isRunningStatus() const {
  return isRunning.load();
}
