/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/9
 * Description: [Provide description here]
 */
#include <Transaction/deadlock_prevention.hpp>
DeadlockPrevention::DeadlockPrevention() {}

bool DeadlockPrevention::acquire_lock(const std::string& transaction_id, const std::string& resource_id) {
  std::lock_guard<std::mutex> lock(graph_mutex);

  // Add an edge in the wait-for graph
  wait_for_graph[transaction_id].push_back(resource_id);

  // Check for potential deadlock
  std::unordered_map<std::string, bool> visited;
  std::unordered_map<std::string, bool> recursion_stack;
  if (has_cycle(transaction_id, visited, recursion_stack)) {
    // If a cycle is detected, remove the edge and return false
    wait_for_graph[transaction_id].pop_back();
    return false;
  }

  return true;
}

void DeadlockPrevention::release_lock(const std::string& transaction_id, const std::string& resource_id) {
  std::lock_guard<std::mutex> lock(graph_mutex);

  // Remove the resource from the wait-for graph
  auto& resources = wait_for_graph[transaction_id];
  resources.erase(std::remove(resources.begin(), resources.end(), resource_id), resources.end());

  // If no more resources are held by the transaction, remove it from the graph
  if (resources.empty()) {
    wait_for_graph.erase(transaction_id);
  }
}

std::optional<std::vector<std::string>> DeadlockPrevention::detect_deadlock(const std::string& transaction_id, const std::string& resource_id) {
  std::lock_guard<std::mutex> lock(graph_mutex);

  // Add the edge temporarily to check for deadlock
  wait_for_graph[transaction_id].push_back(resource_id);

  std::unordered_map<std::string, bool> visited;
  std::unordered_map<std::string, bool> recursion_stack;
  if (has_cycle(transaction_id, visited, recursion_stack)) {
    // If a cycle is detected, return the list of transactions involved in the deadlock
    std::vector<std::string> deadlock_cycle;
    for (const auto& entry : recursion_stack) {
      if (entry.second) {
        deadlock_cycle.push_back(entry.first);
      }
    }
    // Remove the temporary edge
    wait_for_graph[transaction_id].pop_back();
    return deadlock_cycle;
  }

  // Remove the temporary edge if no deadlock is detected
  wait_for_graph[transaction_id].pop_back();
  return std::nullopt;
}

bool DeadlockPrevention::has_cycle(const std::string& transaction_id, std::unordered_map<std::string, bool>& visited, std::unordered_map<std::string, bool>& recursion_stack) {
  if (!visited[transaction_id]) {
    // Mark the current node as visited and add to recursion stack
    visited[transaction_id] = true;
    recursion_stack[transaction_id] = true;

    // Recur for all the vertices adjacent to this vertex
    for (const auto& neighbor : wait_for_graph[transaction_id]) {
      if (!visited[neighbor] && has_cycle(neighbor, visited, recursion_stack)) {
        return true;
      } else if (recursion_stack[neighbor]) {
        return true;
      }
    }
  }

  // Remove the vertex from recursion stack
  recursion_stack[transaction_id] = false;
  return false;
}