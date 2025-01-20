#include <StreamEngine/stream_environment.hpp>
#include <iostream>
#include <fstream>

// Load configuration using ConfigMap
INTELLI::ConfigMap StreamEnvironment::loadConfiguration(const std::string &configFilePath) {
  INTELLI::ConfigMap config;
  if (!config.fromFile(configFilePath)) {
    INTELLI_FATAL_ERROR("Failed to load configuration file: " + configFilePath);
  }
  INTELLI_INFO("Configuration loaded successfully from: " + configFilePath);
  return config;
}

// Define a source stream
std::shared_ptr<DataStream> StreamEnvironment::readSource(const std::string &name, const std::string &sourcePath) {
  auto stream = std::make_shared<DataStream>(name, sourcePath);
  streams.push_back(stream); // Register the stream in the environment
  INTELLI_INFO("Source stream added: " + name + " from path: " + sourcePath);
  return stream;
}

// Execute the pipeline
void StreamEnvironment::execute(const std::string &jobName) {
  INTELLI_INFO("Executing pipeline: " + jobName);

  // Iterate through all registered streams and execute them
  for (auto &stream : streams) {
    INTELLI_INFO("Processing stream: " + stream->getName());
    stream->execute();
  }

  INTELLI_INFO("Pipeline execution completed: " + jobName);
}

// Destructor for cleanup
StreamEnvironment::~StreamEnvironment() {
  INTELLI_INFO("StreamEnvironment destroyed, cleaning up streams.");
  streams.clear();
}
