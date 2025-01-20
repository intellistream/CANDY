#pragma once

#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <StreamEngine/operators.hpp>

using Vector = std::vector<float>;

class DataStream {
public:
  // Constructor
  DataStream(const std::string &name, const std::string &sourcePath);

  // Transformations
  std::shared_ptr<DataStream> filter(std::function<bool(const Vector &)> predicate);
  std::shared_ptr<DataStream> map(std::function<Vector(const Vector &)> mapper);
  std::shared_ptr<DataStream> topK(size_t k, std::function<float(const Vector &)> keyExtractor);
  std::shared_ptr<DataStream> join(const std::shared_ptr<DataStream> &other,
                                   std::function<bool(const Vector &, const Vector &)> joinPredicate);

  // Sink: Write to output
  void writeSink(const std::string &sinkName, const std::string &outputPath);

  // Execution
  void execute();

  // Get the name of the stream
  const std::string &getName() const;

private:
  std::string name;
  std::string sourcePath;
  std::vector<Vector> data;

  void loadData(); // Load data from sourcePath
};
