#pragma once

#include <StreamEngine/data_stream.hpp>
#include <Utils/ConfigMap.hpp>
#include <memory>
#include <string>
#include <vector>

/**
 * @brief StreamEnvironment provides the execution context for defining and running streaming pipelines.
 */
class StreamEnvironment {
public:
  /**
   * @brief Load configuration file into ConfigMap.
   * @param configFilePath The path to the configuration file.
   * @return A populated ConfigMap with configurations.
   */
  INTELLI::ConfigMap loadConfiguration(const std::string &configFilePath);

  /**
   * @brief Define a source stream.
   * @param sourceName Name of the source stream.
   * @param path Path to the source data.
   * @return A shared pointer to the created DataStream.
   */
  std::shared_ptr<DataStream> readSource(const std::string &sourceName, const std::string &path);

  /**
   * @brief Execute the defined streaming pipeline.
   * @param jobName Name of the job/pipeline being executed.
   */
  void execute(const std::string &jobName);

  /**
   * @brief Destructor for cleanup.
   */
  ~StreamEnvironment();

private:
  std::vector<std::shared_ptr<DataStream>> streams; ///< Registered streams in the pipeline
};
