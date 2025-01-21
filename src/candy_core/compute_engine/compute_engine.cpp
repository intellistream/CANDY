#include <candy_core/compute_engine/compute_engine.hpp>

#include <candy_core/compute_engine/compute_engine.hpp>

namespace candy {

void ComputeEngine::validateEqualSize(
    const std::shared_ptr<VectorRecord> &record1,
    const std::shared_ptr<VectorRecord> &record2) {
  if (record1->data->size() != record2->data->size()) {
    throw std::invalid_argument(
        "VectorRecords must have data of the same size.");
  }
}

double ComputeEngine::calculateSimilarity(
    const std::shared_ptr<VectorRecord> &record1,
    const std::shared_ptr<VectorRecord> &record2) {
  validateEqualSize(record1, record2);

  const auto &vec1 = *record1->data;
  const auto &vec2 = *record2->data;

  double dotProduct =
      std::inner_product(vec1.begin(), vec1.end(), vec2.begin(), 0.0);
  double magnitude1 = std::sqrt(
      std::inner_product(vec1.begin(), vec1.end(), vec1.begin(), 0.0));
  double magnitude2 = std::sqrt(
      std::inner_product(vec2.begin(), vec2.end(), vec2.begin(), 0.0));

  if (magnitude1 == 0.0 || magnitude2 == 0.0) {
    throw std::runtime_error("One or both VectorRecords have zero magnitude.");
  }

  return dotProduct / (magnitude1 * magnitude2);
}

double ComputeEngine::computeEuclideanDistance(
    const std::shared_ptr<VectorRecord> &record1,
    const std::shared_ptr<VectorRecord> &record2) {
  validateEqualSize(record1, record2);

  const auto &vec1 = *record1->data;
  const auto &vec2 = *record2->data;

  double sumOfSquares = 0.0;
  for (size_t i = 0; i < vec1.size(); ++i) {
    double diff = vec1[i] - vec2[i];
    sumOfSquares += diff * diff;
  }

  return std::sqrt(sumOfSquares);
}

std::shared_ptr<VectorRecord>
ComputeEngine::normalizeVector(const std::shared_ptr<VectorRecord> &record) {
  const auto &vec = *record->data;
  double magnitude =
      std::sqrt(std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0));

  if (magnitude == 0.0) {
    throw std::runtime_error(
        "Cannot normalize a VectorRecord with zero-magnitude data.");
  }

  VectorData normalized(vec.size());
  std::transform(vec.begin(), vec.end(), normalized.begin(),
                 [&](double val) { return val / magnitude; });

  return std::make_shared<VectorRecord>(record->id, std::move(normalized),
                                        record->timestamp);
}

std::vector<std::shared_ptr<VectorRecord>> ComputeEngine::findTopK(
    const std::vector<std::shared_ptr<VectorRecord>> &records, size_t k,
    std::function<double(const std::shared_ptr<VectorRecord> &)> scorer) {
  if (k > records.size()) {
    throw std::invalid_argument(
        "k cannot be greater than the number of records.");
  }

  std::vector<std::pair<double, std::shared_ptr<VectorRecord>>> scoredRecords;
  scoredRecords.reserve(records.size());

  for (const auto &record : records) {
    scoredRecords.emplace_back(scorer(record), record);
  }

  std::nth_element(
      scoredRecords.begin(), scoredRecords.begin() + k, scoredRecords.end(),
      [](const auto &a, const auto &b) { return a.first > b.first; });

  std::vector<std::shared_ptr<VectorRecord>> topK(k);
  std::transform(scoredRecords.begin(), scoredRecords.begin() + k, topK.begin(),
                 [](const auto &pair) { return pair.second; });

  return topK;
}
} // namespace candy