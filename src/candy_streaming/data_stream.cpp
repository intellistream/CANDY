#include <candy_streaming/data_stream.hpp>

namespace candy {

std::shared_ptr<DataStream> DataStream::filter(FilterFunction filterFunc) {
  auto newStream = std::make_shared<DataStream>(name + "_filter");
  transformations.push_back([this, newStream, filterFunc]() {
    while (!records.empty()) {
      const auto &record = records.front();
      if (filterFunc(record)) {
        newStream->addRecord(record);
      }
      records.pop();
    }
  });
  return newStream;
}

std::shared_ptr<DataStream> DataStream::map(MapFunction mapFunc) {
  auto newStream = std::make_shared<DataStream>(name + "_map");
  transformations.push_back([this, newStream, mapFunc]() {
    while (!records.empty()) {
      const auto &record = records.front();
      newStream->addRecord(mapFunc(record));
      records.pop();
    }
  });
  return newStream;
}

std::shared_ptr<DataStream> DataStream::join(std::shared_ptr<DataStream> otherStream,
                                             JoinFunction joinFunc) {
  auto newStream = std::make_shared<DataStream>(name + "_join");
  transformations.push_back([this, newStream, otherStream, joinFunc]() {
    while (!records.empty()) {
      const auto &leftRecord = records.front();

      // Transform otherStream->records queue into a container for iteration
      std::queue<std::shared_ptr<VectorRecord>> otherRecords = otherStream->records;
      while (!otherRecords.empty()) {
        const auto &rightRecord = otherRecords.front();
        if (joinFunc(leftRecord, rightRecord)) {
          newStream->addRecord(leftRecord); // Example: Emit leftRecord on match
        }
        otherRecords.pop();
      }
      records.pop();
    }
  });
  return newStream;
}

void DataStream::writeSink(const std::string &sinkName, SinkFunction sinkFunc) {
  transformations.push_back([this, sinkFunc]() {
    while (!records.empty()) {
      sinkFunc(records.front());
      records.pop();
    }
  });
}

void DataStream::addRecord(const std::shared_ptr<VectorRecord> &record) {
  records.push(record);
}

void DataStream::processStream() {
  executeTransformations();
}

void DataStream::executeTransformations() {
  for (const auto &transformation : transformations) {
    transformation();
  }
  transformations.clear();
}

} // namespace candy
