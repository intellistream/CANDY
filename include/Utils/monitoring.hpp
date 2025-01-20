#pragma once
#include <vector>
#include <string>
#include <thread>
#include <atomic>

class PerformanceMonitor {
public:
  PerformanceMonitor();
  ~PerformanceMonitor();

  // Start monitoring performance
  void start();

  // Stop monitoring performance
  void stop();

  // Generate a performance report
  void report() const;

private:
  // Atomic flag to control monitoring thread
  std::atomic<bool> monitoringActive;

  // Monitoring thread
  std::thread monitoringThread;

  // Internal method to collect performance data
  void collectMetrics();

  // File path for gperftools CPU profiler
  std::string profileFile;
};
