#include <StreamEngine/stream_environment.hpp>
#include <Utils/ConfigMap.hpp>
#include <Utils/Logging.hpp>
#include <Utils/monitoring.hpp>
#include <ComputeEngine/compute_engine.hpp> // For similarity calculations
#include <iostream>
#include <string>

#define CONFIG_PATH "./config/" // Configuration directory

using namespace std;

void definePipeline(StreamEnvironment &env, const string &configFilePath) {
    // Load configuration
    auto conf = env.loadConfiguration(configFilePath);

    // Define the source stream
    auto sourceStream = env.readSource("VectorSource", conf.getString("inputPath"));

    // Apply filter operation
    auto filteredStream = sourceStream->filter([](const Vector &vec) {
        return vec[0] > 0.5;  // Example filter logic: Keep vectors with the first value > 0.5
    });

    // Apply top-K operation
    auto topKStream = filteredStream->topK(conf.getU64("topK"), [](const Vector &vec) {
        return vec[0];  // Sorting by the first dimension
    });

    // Apply join operation using ComputeEngine's similarity function
    auto joinedStream = topKStream->join(sourceStream, [](const Vector &left, const Vector &right) {
        return ComputeEngine::calculateSimilarity(left, right) > 0.8;  // Threshold for similarity
    });

    // Write results to the sink
    joinedStream->writeSink("VectorSink", conf.getString("outputPath"));

    // Execute the pipeline
    env.execute("Vector Processing Pipeline");
}

int main(int argc, char *argv[]) {
    // Initialize logging
    setupLogging("benchmark.log", LOG_INFO);

    if (argc < 2) {
        INTELLI_ERROR(string("Usage: ") + argv[0] + " <config_file_name>");
        return 1;
    }

    // Construct configuration file path
    string configFilePath = CONFIG_PATH + string(argv[1]);

    // Initialize StreamEnvironment
    StreamEnvironment env;

    // Start performance monitoring
    PerformanceMonitor monitor;
    monitor.start();

    // Define and execute the pipeline
    definePipeline(env, configFilePath);

    // Stop performance monitoring and report results
    monitor.stop();
    monitor.report();

    return 0;
}
