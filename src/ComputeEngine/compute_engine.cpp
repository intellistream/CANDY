#include <ComputeEngine/compute_engine.hpp>
#include <cmath>
#include <numeric>

namespace ComputeEngine {

float calculateSimilarity(const Vector &vec1, const Vector &vec2) {
  return cosineSimilarity(vec1, vec2); // Default to cosine similarity
}

float euclideanDistance(const Vector &vec1, const Vector &vec2) {
  float sum = 0.0;
  for (size_t i = 0; i < vec1.size(); ++i) {
    float diff = vec1[i] - vec2[i];
    sum += diff * diff;
  }
  return std::sqrt(sum);
}

float cosineSimilarity(const Vector &vec1, const Vector &vec2) {
  float dotProduct = 0.0;
  float normA = 0.0;
  float normB = 0.0;
  for (size_t i = 0; i < vec1.size(); ++i) {
    dotProduct += vec1[i] * vec2[i];
    normA += vec1[i] * vec1[i];
    normB += vec2[i] * vec2[i];
  }
  return dotProduct / (std::sqrt(normA) * std::sqrt(normB));
}

} // namespace ComputeEngine
