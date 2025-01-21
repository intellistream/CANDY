#include <candy_core/common/data_types.hpp>
#include <candy_core/compute_engine/compute_engine.hpp>
#include <candy_core/utils/config_map.hpp>
#include <candy_core/utils/logging.hpp>
#include <candy_core/utils/monitoring.hpp>
#include <candy_streaming/stream_environment.hpp>
#include <candy_vector_db/vector_database.hpp>

#include <candy_runtime/operators/log_operator.hpp>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
using namespace candy;

const std::string candy_path = CANDY_PATH; // Defined during build
#define CONFIG_DIR "/config/" // Relative path to configuration files

namespace candy {

// Function to validate configuration
void validateConfiguration(const INTELLI::config_map &conf) {
  if (!conf.existString("inputPath") || !conf.existString("outputPath")) {
    throw runtime_error(
        "Missing required configuration keys: inputPath or outputPath.");
  }
  if (!conf.existU64("topK")) {
    throw runtime_error("Missing required configuration key: topK.");
  }
  if (!conf.existDouble("similarityThreshold")) {
    throw runtime_error(
        "Missing required configuration key: similarityThreshold.");
  }
}

// Function to set up and run the pipeline
void setupAndRunPipeline(const std::string &configFilePath) {
  // Initialize logging
  INTELLI_INFO("Initializing pipeline with configuration file: " +
               configFilePath);

  // Create the streaming environment
  candy::StreamEnvironment env;

  // Load configuration
  auto conf = env.loadConfiguration(configFilePath);

  // Validate configuration
  try {
    validateConfiguration(conf);
    INTELLI_INFO("Configuration validation successful.");
  } catch (const exception &e) {
    INTELLI_ERROR("Configuration validation failed: " + string(e.what()));
    throw;
  }

  // Start performance monitoring
  PerformanceMonitor monitor;
  monitor.start_profiling();
  INTELLI_INFO("Performance monitoring started.");

  try {
    // Define pipeline using the fluent API
    auto sourceStream =
        env.readSource("VectorSource", conf.getString("inputPath"));

    sourceStream
        ->filter([](const std::shared_ptr<VectorRecord> &record) {
          return record && record->data && !record->data->empty() &&
                 (*record->data)[0] > 0.5; // Filter by first value
        })
        ->map([](const std::shared_ptr<VectorRecord> &record) {
          // Normalize the vector using ComputeEngine::normalizeVector
          return ComputeEngine::normalizeVector(record);
        })
        ->join(env.readSource("OtherSource", conf.getString("inputPath")),
               [&](const std::shared_ptr<VectorRecord> &left,
                   const std::shared_ptr<VectorRecord> &right) {
                 // Calculate similarity using
                 // ComputeEngine::calculateSimilarity
                 return ComputeEngine::calculateSimilarity(left, right) >
                        conf.getDouble("similarityThreshold");
               })
        ->writeSink("VectorSink",
                    [](const std::shared_ptr<VectorRecord> &record) {
                      // Log the record's ID when writing to the sink
                      if (record) {
                        INTELLI_INFO("Writing record ID: " + record->id);
                      } else {
                        INTELLI_ERROR("Null record encountered in sink.");
                      }
                    });
    INTELLI_INFO("Pipeline defined successfully.");

    // Add a logging operator for debugging
    auto logOperator = std::make_shared<LogOperator>();
    sourceStream->map(
        [logOperator](const std::shared_ptr<VectorRecord> &record) {
          logOperator->process(record); // Apply the logging operator
          return record;                // Pass the record downstream
        });

    INTELLI_INFO("Log operator added to the pipeline.");

    // Execute the pipeline
    env.execute("Comprehensive Vector Processing Pipeline");
    INTELLI_INFO("Pipeline execution completed.");
  } catch (const std::exception &e) {
    INTELLI_ERROR("Pipeline execution failed: " + std::string(e.what()));
    throw;
  }

  // Stop performance monitoring and report results
  monitor.stop_profiling();

  INTELLI_INFO("Performance monitoring stopped and reported.");
}

} // namespace candy

int main(int argc, char *argv[]) {
  // Default configuration file
  const std::string defaultConfigFile =
      candy_path + CONFIG_DIR + "default_config.txt";

  // Determine configuration file path
  string configFilePath;
  if (argc < 2) {
    INTELLI_WARNING("No configuration file provided. Using default: " +
                    defaultConfigFile);
    configFilePath = defaultConfigFile;
  } else {
    configFilePath = candy_path + CONFIG_DIR + string(argv[1]);
  }

  try {
    // Set up and run the pipeline
    setupAndRunPipeline(configFilePath);
  } catch (const exception &e) {
    INTELLI_ERROR("Fatal error: " + string(e.what()));
    cerr << "Error: " << e.what() << endl;
    return 1;
  }

  INTELLI_INFO("Program finished successfully.");
  return 0;
}
