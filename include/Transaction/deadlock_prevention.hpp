/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/9
 * Description: [Provide description here]
 */
#ifndef INTELLISTREAM_SRC_CONCURRENCY_DEADLOCK_PREVENTION_HPP_
#define INTELLISTREAM_SRC_CONCURRENCY_DEADLOCK_PREVENTION_HPP_
#include <algorithm>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

class DeadlockPrevention {
 public:
  DeadlockPrevention();

  // Register a resource and lock it with a unique id
  bool acquire_lock(const std::string& transaction_id,
                    const std::string& resource_id);

  // Release the resource for a specific transaction
  void release_lock(const std::string& transaction_id,
                    const std::string& resource_id);

  // Detect if a deadlock might occur when acquiring a new lock
  std::optional<std::vector<std::string>> detect_deadlock(
      const std::string& transaction_id, const std::string& resource_id);

 private:
  // Graph to represent resource allocation and waiting
  std::unordered_map<std::string, std::vector<std::string>> wait_for_graph;

  // Mutex for thread safety
  std::mutex graph_mutex;

  // Helper function to perform cycle detection in the wait-for graph
  bool has_cycle(const std::string& transaction_id,
                 std::unordered_map<std::string, bool>& visited,
                 std::unordered_map<std::string, bool>& recursion_stack);
};

#endif  //INTELLISTREAM_SRC_CONCURRENCY_DEADLOCK_PREVENTION_HPP_
