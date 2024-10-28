/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/9
 * Description: [Provide description here]
 */
#include <Performance/monitoring.hpp> // Performance utilities
#include <iostream>
#include <chrono>
#include <string>
#include <Core/vector_db.hpp>

using namespace std;
using namespace chrono;

#include <torch/torch.h>
#include <iostream>
#include <chrono>

using namespace std;
using namespace std::chrono;

// Function to run a specific benchmark scenario
void benchmarkScenario(VectorDB &db, const string &scenario_name) {
  INTELLI_INFO("Running benchmark for: " << scenario_name)
  auto start = high_resolution_clock::now();

  // Run the benchmark scenario (e.g., Insert, Query)
  if (scenario_name == "insert") {
    for (int i = 0; i < 10000; ++i) {
      torch::Tensor new_tensor = torch::tensor({
        static_cast<float>(i), static_cast<float>(i + 1), static_cast<float>(i + 2)
      });
      db.insert_tensor(new_tensor);
    }
  } else if (scenario_name == "query") {
    for (int i = 0; i < 1000; ++i) {
      torch::Tensor query_tensor = torch::tensor({
        static_cast<float>(i), static_cast<float>(i + 1), static_cast<float>(i + 2)
      });
      auto result = db.query_nearest_tensors(query_tensor, 5);
      // Simulate some usage of the result
      (void) result;
    }
  }

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  INTELLI_INFO("Time taken for " << scenario_name << ": " << duration.count() << " ms");
}

// Function to run a series of benchmarks
void runBenchmarks(VectorDB &db) {
  INTELLI_INFO("Starting benchmark tests...");

  // Benchmark different scenarios
  benchmarkScenario(db, "insert");
  benchmarkScenario(db, "query");

  INTELLI_INFO("Benchmark tests completed.");
}

int main() {
  // Initialize torch
  torch::manual_seed(0);

  // Initialize the Vector Database with 3 dimensions and default search algorithm
  VectorDB db(3);

  // Set up performance monitoring (e.g., CPU, memory)
  PerformanceMonitor monitor;
  monitor.start();

  // Run the benchmarks
  runBenchmarks(db);

  // Stop performance monitoring and show results
  monitor.stop();
  monitor.report();

  return 0;
}
