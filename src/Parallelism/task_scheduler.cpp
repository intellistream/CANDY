/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/11
 * Description: [Provide description here]
 */
#include <Parallelism/task_scheduler.hpp>

#include <thread>
#include <vector>
#include <iostream>

// Concrete implementation of TaskScheduler for multi-threaded execution
class ThreadPoolTaskScheduler : public TaskScheduler {
 public:
  ThreadPoolTaskScheduler(size_t num_threads = std::thread::hardware_concurrency())
      : num_threads_(num_threads) {}

  // Implementation of parallel_for using multiple threads
  void parallel_for(size_t start, size_t end, const std::function<void(size_t)> &func) const override {
    size_t range = end - start;
    size_t chunk_size = (range + num_threads_ - 1) / num_threads_;

    std::vector<std::thread> threads;
    for (size_t i = 0; i < num_threads_; ++i) {
      size_t chunk_start = start + i * chunk_size;
      size_t chunk_end = std::min(chunk_start + chunk_size, end);

      if (chunk_start >= end) {
        break;
      }

      threads.emplace_back([chunk_start, chunk_end, &func]() {
        for (size_t j = chunk_start; j < chunk_end; ++j) {
          func(j);
        }
      });
    }

    for (auto &thread : threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }
  }

 private:
  size_t num_threads_;
};