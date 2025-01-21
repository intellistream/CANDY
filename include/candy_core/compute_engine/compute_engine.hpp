// ComputeEngine.hpp
#ifndef CANDY_CORE_COMPUTE_ENGINE_HPP
#define CANDY_CORE_COMPUTE_ENGINE_HPP

#include <algorithm>
#include <candy_core/common/data_types.hpp>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace candy {

class ComputeEngine {
public:
  // Calculate cosine similarity between two VectorRecords
  static double
  calculateSimilarity(const std::shared_ptr<VectorRecord> &record1,
                      const std::shared_ptr<VectorRecord> &record2);

  // Compute Euclidean distance between two VectorRecords
  static double
  computeEuclideanDistance(const std::shared_ptr<VectorRecord> &record1,
                           const std::shared_ptr<VectorRecord> &record2);

  // Normalize the data in a VectorRecord
  static std::shared_ptr<VectorRecord>
  normalizeVector(const std::shared_ptr<VectorRecord> &record);

  // Find top-K VectorRecords based on a scoring function
  static std::vector<std::shared_ptr<VectorRecord>>
  findTopK(const std::vector<std::shared_ptr<VectorRecord>> &records, size_t k,
           std::function<double(const std::shared_ptr<VectorRecord> &)> scorer);

  // Validate if two VectorRecords have data of the same size
  static void validateEqualSize(const std::shared_ptr<VectorRecord> &record1,
                                const std::shared_ptr<VectorRecord> &record2);

private:
  ComputeEngine() = delete; // Prevent instantiation
};

} // namespace candy

#endif // CANDY_CORE_COMPUTE_ENGINE_HPP