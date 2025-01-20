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
  struct Metrics {
    uint64_t cpuCycles;
    uint64_t instructions;
    uint64_t cacheMisses;
  };

  // Atomic flag to control monitoring thread
  std::atomic<bool> monitoringActive;

  // Collected metrics
  std::vector<Metrics> collectedMetrics;

  // Monitoring thread
  std::thread monitoringThread;

  // Internal method to collect performance data
  void collectMetrics();

  // Open a perf event
  int openPerfEvent(int type, int config) const;

  // Read data from a perf event file descriptor
  uint64_t readPerfEvent(int fd) const;
};
