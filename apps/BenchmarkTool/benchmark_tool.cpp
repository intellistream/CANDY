/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/9
 * Description: [Provide description here]
 */
#include <Performance/monitoring.hpp>  // Performance utilities
#include <Utils/logging.hpp>
#include "scenarios.hpp"

#include <chrono>
#include <iostream>
#include <map>
#include <string>

using namespace std;
using namespace chrono;

#define CONFIG_PATH \
  "./config/"  // Common configuration directory in the current project's root

// Function to run a specific benchmark scenario
void benchmarkScenario(VectorDB& db, ScenarioConfig& conf) {

  auto it = scenarios.find(conf.scenario_name);

  if (it == scenarios.end()) {
    INTELLI_DEBUG("Scenario not found: " << conf.scenario_name);
    exit(EXIT_FAILURE);
  }
  INTELLI_INFO("Running benchmark for: " + conf.scenario_name);
  auto start = high_resolution_clock::now();

  it->second(db, conf);

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  INTELLI_INFO(string("Time taken for ") + conf.scenario_name + ": " +
               to_string(duration.count()) + " ms");
}

// Function to run a series of benchmarks
void runBenchmarks(VectorDB& db, ScenarioConfig& conf) {
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
    INTELLI_ERROR(string("Usage: ") + argv[0] + " <config_file_name>");
    return 1;
  }

  // Construct the full path to the configuration file
  string configFilePath = CONFIG_PATH + string(argv[1]);
  ScenarioConfig conf(configFilePath);

  // Check if the configuration file can be read successfully
  if (!conf.isValid()) {
    INTELLI_ERROR("Failed to read configuration file: " + configFilePath);
    return 1;
  }

  // Initialize the Vector Database with 3 dimensions and default search algorithm
  VectorDB db(conf.dimension);

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
