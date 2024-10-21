/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/9
 * Description: [Provide description here]
 */
#include <Performance/monitoring.hpp> // Performance utilities
#include "scenarios.hpp"
#include <Utils/logging.hpp>

#include <Algorithms/hnsw.hpp>

#include <iostream>
#include <chrono>
#include <map>
#include <string>

using namespace std;
using namespace chrono;

// Function to run a specific benchmark scenario
void benchmarkScenario(VectorDB &db, ScenarioConfig &conf) {

  auto it = scenarios.find(conf.scenario_name);
  if (it == scenarios.end()) { 
    cout << "Scenario not found: " << conf.scenario_name <<endl;
    exit(EXIT_FAILURE);
  }
  INTELLI_INFO("Running benchmark for: " + conf.scenario_name);
  auto start = high_resolution_clock::now();
  
  it->second(db, conf); 
  
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  INTELLI_INFO(string("Time taken for ") + conf.scenario_name + 
    ": " + to_string(duration.count()) + " ms");
}

// Function to run a series of benchmarks
void runBenchmarks(VectorDB &db, ScenarioConfig &conf) {
  INTELLI_INFO("Starting benchmark tests...");  

  // Benchmark different scenarios
  benchmarkScenario(db, conf);

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
  ScenarioConfig conf(argv[1]);

  // Initialize the Vector Database with 3 dimensions and default search algorithm
  std::shared_ptr<SearchAlgorithm> hnsw_algorithm = std::make_shared<HNSWAlgorithm>();
  VectorDB db(conf.dimension, hnsw_algorithm);

  // Set up performance monitoring (e.g., CPU, memory)
  PerformanceMonitor monitor;
  monitor.start();

  // Run the benchmarks
  runBenchmarks(db, conf);

  // Stop performance monitoring and show results
  monitor.stop();
  monitor.report();

  return 0;
}