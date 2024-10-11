/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/9
 * Description: [Provide description here]
 */
#ifndef INTELLISTREAM_SRC_PERFORMANCE_MONITORING_HPP_
#define INTELLISTREAM_SRC_PERFORMANCE_MONITORING_HPP_

#include <iostream>
#include <chrono>
#include <thread>
#include <atomic>

class PerformanceMonitor {
 public:
  // Constructor
  PerformanceMonitor() : monitoring(false), cpu_usage(0.0), memory_usage(0.0) {}

  // Start monitoring performance metrics
  void start() {
    monitoring = true;
    monitor_thread = std::thread(&PerformanceMonitor::monitor, this);
  }

  // Stop monitoring performance metrics
  void stop() {
    monitoring = false;
    if (monitor_thread.joinable()) {
      monitor_thread.join();
    }
  }

  // Report the gathered metrics
  void report() const {
    std::cout << "Performance Report:" << std::endl;
    std::cout << "CPU Usage (approximate): " << cpu_usage << "%" << std::endl;
    std::cout << "Memory Usage (approximate): " << memory_usage << " MB" << std::endl;
  }

 private:
  std::atomic<bool> monitoring;
  std::thread monitor_thread;
  double cpu_usage;
  double memory_usage;

  // Function to simulate monitoring of CPU and memory usage
  void monitor() {
    while (monitoring) {
      // Simulate CPU usage monitoring (this would be replaced with actual system calls in a real implementation)
      cpu_usage = (rand() % 100) + 1; // Random CPU usage between 1 and 100%

      // Simulate memory usage monitoring (this would be replaced with actual system calls in a real implementation)
      memory_usage = (rand() % 800) + 200; // Random memory usage between 200 and 1000 MB

      // Sleep for a while before the next measurement
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }
};

#endif //INTELLISTREAM_SRC_PERFORMANCE_MONITORING_HPP_
