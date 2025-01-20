#define CATCH_CONFIG_MAIN
#include <catch2/catch_approx.hpp>
#include <ComputeEngine/compute_engine.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
using namespace ComputeEngine;

// Test cases for ComputeEngine
TEST_CASE("ComputeEngine: Euclidean Distance", "[compute]") {
  Vector vec1 = {1.0, 2.0, 3.0};
  Vector vec2 = {4.0, 5.0, 6.0};

  float expectedDistance = std::sqrt(27.0); // √((4-1)² + (5-2)² + (6-3)²)
  REQUIRE(euclideanDistance(vec1, vec2) == Catch::Approx(expectedDistance));
}

TEST_CASE("ComputeEngine: Cosine Similarity", "[compute]") {
  Vector vec1 = {1.0, 0.0, -1.0};
  Vector vec2 = {-1.0, 0.0, 1.0};

  float expectedSimilarity = -1.0; // Cosine of 180° (vectors are opposites)
  REQUIRE(cosineSimilarity(vec1, vec2) == Catch::Approx(expectedSimilarity));
}

TEST_CASE("ComputeEngine: Default Similarity (Cosine)", "[compute]") {
  Vector vec1 = {1.0, 2.0, 3.0};
  Vector vec2 = {4.0, 5.0, 6.0};

  // Expected cosine similarity
  float dotProduct = 1 * 4 + 2 * 5 + 3 * 6;       // 32
  float normA = std::sqrt(1 * 1 + 2 * 2 + 3 * 3); // √14
  float normB = std::sqrt(4 * 4 + 5 * 5 + 6 * 6); // √77
  float expectedSimilarity = dotProduct / (normA * normB);

  REQUIRE(calculateSimilarity(vec1, vec2) == Catch::Approx(expectedSimilarity));
}

TEST_CASE("ComputeEngine: Edge Cases", "[compute]") {
  SECTION("Identical Vectors") {
    Vector vec = {1.0, 2.0, 3.0};
    REQUIRE(euclideanDistance(vec, vec) == Catch::Approx(0.0));
    REQUIRE(cosineSimilarity(vec, vec) == Catch::Approx(1.0));
  }

  SECTION("Zero Vectors for Cosine Similarity") {
    Vector vec1 = {0.0, 0.0, 0.0};
    Vector vec2 = {1.0, 2.0, 3.0};
    REQUIRE(cosineSimilarity(vec1, vec2) ==
            Catch::Approx(0.0)); // Division by zero handled
  }
}
