// data_stream.hpp
#ifndef CANDY_STREAMING_DATA_STREAM_HPP
#define CANDY_STREAMING_DATA_STREAM_HPP
#include <candy_core/common/data_types.hpp>
#include <functional>
#include <memory>
#include <queue>
#include <string>
#include <vector>

namespace candy {

class DataStream {
public:
  using FilterFunction = std::function<bool(const std::shared_ptr<VectorRecord>&)>;
  using MapFunction = std::function<std::shared_ptr<VectorRecord>(const std::shared_ptr<VectorRecord>&)>;
  using JoinFunction =
      std::function<bool(const std::shared_ptr<VectorRecord>&, const std::shared_ptr<VectorRecord>&)>;
  using SinkFunction = std::function<void(const std::shared_ptr<VectorRecord>&)>;

  // Constructor
  explicit DataStream(const std::string &name) : name(name) {}

  // Apply a filter to the stream
  std::shared_ptr<DataStream> filter(FilterFunction filterFunc);

  // Apply a map function to the stream
  std::shared_ptr<DataStream> map(MapFunction mapFunc);

  // Join with another stream
  std::shared_ptr<DataStream> join(std::shared_ptr<DataStream> otherStream,
                                   JoinFunction joinFunc);

  // Write to a sink
  void writeSink(const std::string &sinkName, SinkFunction sinkFunc);

  // Internal: Add data to the stream
  void addRecord(const std::shared_ptr<VectorRecord>& record);

  // Internal: Process the stream
  void processStream();

private:
  std::string name;
  std::queue<std::shared_ptr<VectorRecord>> records;
  std::vector<std::function<void()>> transformations;

  // Execute transformations
  void executeTransformations();
};

} // namespace candy

#endif // CANDY_STREAMING_DATA_STREAM_HPP
