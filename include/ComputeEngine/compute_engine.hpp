#pragma once
#include <vector>
#include <string>

using Vector = std::vector<float>;

namespace ComputeEngine {

/**
 * @brief Computes similarity between two vectors.
 * @param vec1 First vector.
 * @param vec2 Second vector.
 * @return Similarity score.
 */
float calculateSimilarity(const Vector &vec1, const Vector &vec2);

/**
 * @brief Computes the Euclidean distance between two vectors.
 * @param vec1 First vector.
 * @param vec2 Second vector.
 * @return Distance value.
 */
float euclideanDistance(const Vector &vec1, const Vector &vec2);

/**
 * @brief Computes the cosine similarity between two vectors.
 * @param vec1 First vector.
 * @param vec2 Second vector.
 * @return Cosine similarity score.
 */
float cosineSimilarity(const Vector &vec1, const Vector &vec2);

} // namespace ComputeEngine
