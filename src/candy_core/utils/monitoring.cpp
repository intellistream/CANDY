#include <candy_core/utils/monitoring.hpp>

namespace candy {

PerformanceMonitor::PerformanceMonitor(const std::string &profileOutput)
    : profileOutputFile(profileOutput), profiling(false) {}

PerformanceMonitor::~PerformanceMonitor() {
  if (profiling) {
    stop_profiling();
  }
}

void PerformanceMonitor::start_profiling() {
  if (!profiling) {
    ProfilerStart(profileOutputFile.c_str());
    profiling = true;
    std::cout << "Profiling started: " << profileOutputFile << std::endl;
  } else {
    std::cerr << "Profiling is already running." << std::endl;
  }
}

void PerformanceMonitor::stop_profiling() {
  if (profiling) {
    ProfilerStop();
    profiling = false;
    std::cout << "Profiling stopped and saved to: " << profileOutputFile
              << std::endl;
  } else {
    std::cerr << "Profiling is not running." << std::endl;
  }
}

void PerformanceMonitor::start_timer() {
  startTime = std::chrono::high_resolution_clock::now();
  std::cout << "Timer started." << std::endl;
}

void PerformanceMonitor::stop_timer(const std::string &taskName) {
  auto endTime = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime)
          .count();
  std::cout << "Task [" << taskName << "] completed in " << duration << " ms."
            << std::endl;
}

} // namespace candy
