/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/11
 * Description: [Provide description here]
 */
#ifndef CANDY_SRC_CONCURRENCY_TASK_SCHEDULER_HPP_
#define CANDY_SRC_CONCURRENCY_TASK_SCHEDULER_HPP_

#include <functional>
#include <cstddef>

// Abstract class defining the interface for task scheduling
class TaskScheduler {
 public:
  // Virtual destructor for proper cleanup in derived classes
  virtual ~TaskScheduler() = default;

  // Pure virtual function for parallel execution of a task over a range
  virtual void parallel_for(size_t start, size_t end, const std::function<void(size_t)>& func) const = 0;
};
#endif //CANDY_SRC_CONCURRENCY_TASK_SCHEDULER_HPP_
