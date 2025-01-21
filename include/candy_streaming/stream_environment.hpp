#ifndef STREAM_ENVIRONMENT_HPP
#define STREAM_ENVIRONMENT_HPP

#include <candy_core/utils/config_map.hpp>
#include <candy_runtime/operators/base_operators.hpp>
#include <candy_streaming/data_stream.hpp>
#include <memory>
#include <string>
#include <vector>

namespace candy {

class StreamEnvironment {
public:
  // Constructor to initialize the environment
  explicit StreamEnvironment() = default;

  // Load configuration from a file
  INTELLI::config_map loadConfiguration(const std::string &filePath);

  // Read a source and create a DataStream
  std::shared_ptr<DataStream> readSource(const std::string &sourceType, const std::string &path);

  // Execute the pipeline
  void execute(const std::string &pipelineName);

  // Add an operator to the pipeline
  void addOperator(const std::shared_ptr<candy::BaseOperator> &op);

private:
  std::vector<std::shared_ptr<candy::BaseOperator>> operators_;
};

} // namespace candy

#endif // STREAM_ENVIRONMENT_HPP