#include <candy_vector_db/indexing/anns.hpp>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace candy {

void ANNS::insert(const std::shared_ptr<VectorRecord> &record) {
  index[record->id] = record;
}

std::vector<std::shared_ptr<VectorRecord>> ANNS::search(const VectorData &query,
                                                        size_t k) {
  if (k > index.size()) {
    throw std::invalid_argument(
        "k is larger than the number of vectors in the index.");
  }

  std::vector<std::pair<double, std::shared_ptr<VectorRecord>>> scoredResults;

  for (const auto &[id, record] : index) {
    double similarity = std::inner_product(query.begin(), query.end(),
                                           record->data->begin(), 0.0);
    scoredResults.emplace_back(similarity, record);
  }

  std::sort(scoredResults.begin(), scoredResults.end(), std::greater<>());

  std::vector<std::shared_ptr<VectorRecord>> topK;
  for (size_t i = 0; i < k; ++i) {
    topK.push_back(scoredResults[i].second);
  }

  return topK;
}

void ANNS::remove(const std::string &id) { index.erase(id); }

} // namespace candy