/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/9
 * Description: [Provide description here]
 */
#include <Core/vector_db.hpp>
#include <Performance/monitoring.hpp> // Performance utilities

#include <iostream>
#include <chrono>

using namespace std;
using namespace chrono;

// Function to run a specific benchmark scenario
void benchmarkScenario(VectorDB &db, const string &scenario_name) {
  cout << "Running benchmark for: " << scenario_name << endl;
  auto start = high_resolution_clock::now();

  // Run the benchmark scenario (e.g., Insert, Query, Delete)
  if (scenario_name == "insert") {
    for (int i = 0; i < 10000; ++i) {
      vector<float> new_vector = {static_cast<float>(i), static_cast<float>(i + 1), static_cast<float>(i + 2)};
      db.insert_vector(new_vector);
    }
  } else if (scenario_name == "query") {
    for (int i = 0; i < 1000; ++i) {
      vector<float> query_vector = {static_cast<float>(i), static_cast<float>(i + 1), static_cast<float>(i + 2)};
      auto result = db.query_nearest_vectors(query_vector, 5);
      // Simulate some usage of the result
      (void)result;
    }
  }

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  cout << "Time taken for " << scenario_name << ": " << duration.count() << " ms" << endl;
}

// Function to run a series of benchmarks
void runBenchmarks(VectorDB &db) {
  cout << "Starting benchmark tests..." << endl;

  // Benchmark different scenarios
  benchmarkScenario(db, "insert");
  benchmarkScenario(db, "query");

  cout << "Benchmark tests completed." << endl;
}

int main() {
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