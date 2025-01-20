#include <Utils/monitoring.hpp>
#include <Utils/Logging.hpp>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <chrono>

PerformanceMonitor::PerformanceMonitor() : monitoringActive(false) {}

PerformanceMonitor::~PerformanceMonitor() {
    if (monitoringActive.load()) {
        stop();
    }
}

void PerformanceMonitor::start() {
    monitoringActive.store(true);
    collectedMetrics.clear();
    monitoringThread = std::thread(&PerformanceMonitor::collectMetrics, this);
    INTELLI_INFO("Performance monitoring started.");
}

void PerformanceMonitor::stop() {
    if (monitoringActive.load()) {
        monitoringActive.store(false);
        if (monitoringThread.joinable()) {
            monitoringThread.join();
        }
        INTELLI_INFO("Performance monitoring stopped.");
    }
}

void PerformanceMonitor::report() const {
    INTELLI_INFO("Generating performance report...");
    uint64_t totalCycles = 0, totalInstructions = 0, totalCacheMisses = 0;

    for (const auto &metrics : collectedMetrics) {
        totalCycles += metrics.cpuCycles;
        totalInstructions += metrics.instructions;
        totalCacheMisses += metrics.cacheMisses;
    }

    size_t count = collectedMetrics.size();
    if (count > 0) {
        std::cout << "Performance Report:\n";
        std::cout << "  Average CPU Cycles: " << totalCycles / count << "\n";
        std::cout << "  Average Instructions: " << totalInstructions / count << "\n";
        std::cout << "  Average Cache Misses: " << totalCacheMisses / count << "\n";
    } else {
        std::cout << "No data collected.\n";
    }
}

void PerformanceMonitor::collectMetrics() {
    // Open perf events
    int cyclesFd = openPerfEvent(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
    int instructionsFd = openPerfEvent(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
    int cacheMissesFd = openPerfEvent(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES);

    if (cyclesFd < 0 || instructionsFd < 0 || cacheMissesFd < 0) {
        INTELLI_FATAL_ERROR("Failed to open perf events.");
    }

    while (monitoringActive.load()) {
        Metrics metrics;
        metrics.cpuCycles = readPerfEvent(cyclesFd);
        metrics.instructions = readPerfEvent(instructionsFd);
        metrics.cacheMisses = readPerfEvent(cacheMissesFd);
        collectedMetrics.push_back(metrics);

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    close(cyclesFd);
    close(instructionsFd);
    close(cacheMissesFd);
}

int PerformanceMonitor::openPerfEvent(int type, int config) const {
    struct perf_event_attr pe = {};
    pe.type = type;
    pe.size = sizeof(struct perf_event_attr);
    pe.config = config;
    pe.disabled = 0;
    pe.exclude_kernel = 1; // Don't count kernel events
    pe.exclude_hv = 1;     // Don't count hypervisor events

    int fd = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
    if (fd < 0) {
        INTELLI_ERROR("Error opening perf event: " + std::string(strerror(errno)));
    }
    return fd;
}

uint64_t PerformanceMonitor::readPerfEvent(int fd) const {
    uint64_t value = 0;
    if (read(fd, &value, sizeof(uint64_t)) < 0) {
        INTELLI_ERROR("Error reading perf event: " + std::string(strerror(errno)));
    }
    return value;
}
