#include <candy_core/utils/logging.hpp>
#include <candy_streaming/data_stream.hpp>
#include <candy_streaming/stream_environment.hpp>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace candy {

INTELLI::config_map
StreamEnvironment::loadConfiguration(const std::string &filePath) {
  INTELLI::config_map config;
  if (!config.fromFile(filePath)) {
    throw std::runtime_error("Failed to load configuration from: " + filePath);
  }
  return config;
}

std::shared_ptr<candy::DataStream>
StreamEnvironment::readSource(const std::string &sourceType,
                              const std::string &path) {
  return std::make_shared<candy::DataStream>(sourceType);
}

void StreamEnvironment::execute(const std::string &pipelineName) {
  if (operators_.empty()) {
    throw std::runtime_error(
        "Pipeline execution failed: No operators defined.");
  }

  INTELLI_INFO("Executing pipeline: " + pipelineName);
  for (const auto &op : operators_) {
    op->open();
    op->process(nullptr); // Replace with actual data in the pipeline flow
    op->close();
  }
  INTELLI_INFO("Pipeline execution completed.");
}

void StreamEnvironment::addOperator(
    const std::shared_ptr<candy::BaseOperator> &op) {
  operators_.emplace_back(op);
}

} // namespace candy
