#include <Utils/monitoring.hpp>
#include <Utils/Logging.hpp>
#include <gperftools/profiler.h>
#include <iostream>
#include <chrono>

PerformanceMonitor::PerformanceMonitor()
    : monitoringActive(false), profileFile("cpu_profile.prof") {}

PerformanceMonitor::~PerformanceMonitor() {
    if (monitoringActive.load()) {
        stop();
    }
}

void PerformanceMonitor::start() {
    if (monitoringActive.load()) {
        INTELLI_WARNING("Performance monitoring is already active.");
        return;
    }

    monitoringActive.store(true);
    ProfilerStart(profileFile.c_str());
    monitoringThread = std::thread(&PerformanceMonitor::collectMetrics, this);
    INTELLI_INFO("Performance monitoring started.");
}

void PerformanceMonitor::stop() {
    if (!monitoringActive.load()) {
        INTELLI_WARNING("Performance monitoring is not active.");
        return;
    }

    monitoringActive.store(false);
    if (monitoringThread.joinable()) {
        monitoringThread.join();
    }
    ProfilerStop();
    INTELLI_INFO("Performance monitoring stopped. Profile saved to: " + profileFile);
}

void PerformanceMonitor::report() const {
    INTELLI_INFO("Generating performance report...");
    std::cout << "Performance data has been saved to: " << profileFile << "\n";
}

void PerformanceMonitor::collectMetrics() {
    while (monitoringActive.load()) {
        // Simulating periodic data collection for other performance metrics
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}
