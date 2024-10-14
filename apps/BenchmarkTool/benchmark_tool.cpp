/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/9
 * Description: [Provide description here]
 */
#include <Performance/monitoring.hpp> // Performance utilities
#include "scenarios.hpp"
#include "benchmark_config.hpp"
#include <Utils/logging.hpp>

#include <iostream>
#include <chrono>
#include <map>
#include <string>

using namespace std;
using namespace chrono;

// Function to run a specific benchmark scenario
void benchmarkScenario(VectorDB &db, const string &scenario_name) {

  auto it = scenarios.find(scenario_name);
  if (it == scenarios.end()) { 
    cout << "Scenario not found: " << scenario_name <<endl;
  }

  INTELLI_INFO("Running benchmark for: " + scenario_name);
  auto start = high_resolution_clock::now();

  it->second(db);

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  INTELLI_INFO(string("Time taken for ") + scenario_name + ": " + to_string(duration.count()) + " ms");
}

// Function to run a series of benchmarks
void runBenchmarks(VectorDB &db) {
  INTELLI_INFO("Starting benchmark tests...");  

  // Benchmark different scenarios
  benchmarkScenario(db, "insert");
  benchmarkScenario(db, "query");

  INTELLI_INFO("Benchmark tests completed.");
}

int main(int argc, char* argv[]) {

  // Init logging
  setupLogging("benchmark.log", LOG_INFO);

  // Parse the benchmark file
  if (argc < 2) { 
    INTELLI_ERROR(string("Usage: ") + argv[0] + " <config_file_path>");
    return 1;
  }
  BenchmarkConfig conf(argv[1]);

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