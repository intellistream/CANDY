#include <ComputeEngine/compute_engine.hpp> // For similarity calculations
#include <StreamEngine/stream_environment.hpp>
#include <Utils/ConfigMap.hpp>
#include <Utils/Logging.hpp>
#include <Utils/monitoring.hpp>
#include <iostream>
#include <string>

using namespace std;
const std::string candy_path = CANDY_PATH;
#define CONFIG_DIR "/config/" // Configuration directory relative to CANDY_PATH

void validateConfiguration(const INTELLI::ConfigMap &conf) {
  if (!conf.existString("inputPath") || !conf.existString("outputPath")) {
    throw runtime_error(
        "Missing required configuration keys: inputPath or outputPath.");
  }

  if (!conf.existU64("topK")) {
    throw runtime_error("Missing required configuration key: topK.");
  }
}

void definePipeline(StreamEnvironment &env, const string &configFilePath) {
  // Load configuration
  auto conf = env.loadConfiguration(configFilePath);

  // Validate configuration
  validateConfiguration(conf);

  // Define the source stream
  auto sourceStream =
      env.readSource("VectorSource", conf.getString("inputPath"));

  // Apply filter operation
  auto filteredStream = sourceStream->filter([](const Vector &vec) {
    return vec[0] >
           0.5; // Example filter logic: Keep vectors with the first value > 0.5
  });

  // Apply top-K operation
  auto topKStream =
      filteredStream->topK(conf.getU64("topK"), [](const Vector &vec) {
        return vec[0]; // Sorting by the first dimension
      });

  // Apply join operation using ComputeEngine's similarity function
  auto joinedStream = topKStream->join(
      sourceStream, [](const Vector &left, const Vector &right) {
        return ComputeEngine::calculateSimilarity(left, right) >
               0.8; // Threshold for similarity
      });

  // Write results to the sink
  joinedStream->writeSink("VectorSink", conf.getString("outputPath"));

  // Execute the pipeline
  env.execute("Vector Processing Pipeline");
}

int main(int argc, char *argv[]) {
  // Initialize logging
  setupLogging("benchmark.log", LOG_INFO);

  // Default configuration file
  const std::string candy_path = CANDY_PATH; // Path defined via -DCANDY_PATH
  const std::string defaultConfigFile =
      candy_path + CONFIG_DIR + "default_config.txt";

  // Check for command-line arguments
  string configFilePath;
  if (argc < 2) {
    INTELLI_WARNING("No configuration file provided. Using default: " +
                    defaultConfigFile);
    configFilePath = defaultConfigFile;
  } else {
    configFilePath = candy_path + CONFIG_DIR + string(argv[1]);
  }

  // Initialize StreamEnvironment
  StreamEnvironment env;

  // Start performance monitoring
  PerformanceMonitor monitor;
  monitor.start();

  try {
    // Define and execute the pipeline
    definePipeline(env, configFilePath);
  } catch (const exception &e) {
    INTELLI_ERROR("Pipeline execution failed: " + string(e.what()));
    return 1;
  }

  // Stop performance monitoring and report results
  monitor.stop();
  monitor.report();

  return 0;
}