#pragma once

#include <vector>
#include <functional>
#include <string>

using Vector = std::vector<float>;

namespace Operator {

// Filter operation: Retains vectors that satisfy a given predicate
void applyFilter(const std::vector<Vector> &input,
                 std::function<bool(const Vector &)> predicate,
                 std::vector<Vector> &output);

// Top-K operation: Finds top K vectors based on a key extractor
void applyTopK(const std::vector<Vector> &input,
               size_t k,
               std::function<float(const Vector &)> keyExtractor,
               std::vector<Vector> &output);

// Join operation: Joins two streams based on a predicate
void applyJoin(const std::vector<Vector> &left,
               const std::vector<Vector> &right,
               std::function<bool(const Vector &, const Vector &)> joinPredicate,
               std::vector<Vector> &output);

// Sink: Writes the data to a file
void writeToSink(const std::vector<Vector> &data, const std::string &filePath);

} // namespace Operator
