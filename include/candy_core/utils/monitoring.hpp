#ifndef CANDY_CORE_MONITOR_HPP
#define CANDY_CORE_MONITOR_HPP

#include <chrono>
#include <fstream>
#include <gperftools/profiler.h>
#include <iostream>
#include <string>

namespace candy {

class PerformanceMonitor {
public:
  PerformanceMonitor(const std::string &profileOutput = "profile.prof");
  ~PerformanceMonitor();

  // Start profiling
  void start_profiling();

  // Stop profiling and save the results
  void stop_profiling();

  // Start the timer for measuring elapsed time
  void start_timer();

  // Stop the timer and print elapsed time
  void stop_timer(const std::string &taskName);

private:
  std::string profileOutputFile;
  std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
  bool profiling;
};

} // namespace candy
#endif // CANDY_CORE_MONITOR_HPP