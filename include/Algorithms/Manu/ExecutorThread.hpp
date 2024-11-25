//
// Created by zhonghao on 25/11/24.
//

#ifndef EXECUTOR_THREAD_HPP
#define EXECUTOR_THREAD_HPP

#include <thread>
#include <atomic>

class ExecutorThread {
protected:
  std::atomic<bool> isRunning;
  std::thread workerThread;

public:
  ExecutorThread();
  virtual ~ExecutorThread();

  void start();
  void stop();
  bool isRunningStatus() const;

protected:
  virtual void execute() = 0; // Pure virtual function for thread logic.
};

#endif
