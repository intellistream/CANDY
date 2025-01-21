#define CATCH_CONFIG_MAIN
#include <candy_core/common/data_types.hpp>
#include <candy_core/compute_engine/compute_engine.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>

TEST_CASE("ComputeEngine calculates cosine similarity correctly", "[compute_similarity]") {
  auto vec1 = candy::create_vector_record("id1", {1.0, 2.0, 3.0}, 1);
  auto vec2 = candy::create_vector_record("id2", {4.0, 5.0, 6.0}, 2);

  double similarity = candy::ComputeEngine::calculateSimilarity(vec1, vec2);

  REQUIRE(similarity == Catch::Approx(0.974631846).epsilon(0.0001));
}

TEST_CASE("ComputeEngine calculates Euclidean distance correctly", "[compute_distance]") {
  auto vec1 = candy::create_vector_record("id1", {1.0, 2.0, 3.0}, 1);
  auto vec2 = candy::create_vector_record("id2", {4.0, 5.0, 6.0}, 2);

  double distance = candy::ComputeEngine::computeEuclideanDistance(vec1, vec2);

  REQUIRE(distance == Catch::Approx(5.196152422).epsilon(0.0001));
}

TEST_CASE("ComputeEngine normalizes a vector correctly", "[normalize_vector]") {
  auto vec = candy::create_vector_record("id1", {3.0, 4.0}, 1);
  auto normalized = candy::ComputeEngine::normalizeVector(vec);

  REQUIRE(normalized->data->size() == vec->data->size());
  REQUIRE((*normalized->data)[0] == Catch::Approx(0.6).epsilon(0.0001));
  REQUIRE((*normalized->data)[1] == Catch::Approx(0.8).epsilon(0.0001));
}

TEST_CASE("ComputeEngine finds top K correctly", "[find_top_k]") {
  std::vector<std::shared_ptr<candy::VectorRecord>> records = {
    candy::create_vector_record("id1", {1.0, 2.0}, 1),
    candy::create_vector_record("id2", {3.0, 4.0}, 2),
    candy::create_vector_record("id3", {5.0, 6.0}, 3)};

  auto topK = candy::ComputeEngine::findTopK(records, 2, [](const std::shared_ptr<candy::VectorRecord>& record) {
    return record->timestamp;
  });

  REQUIRE(topK.size() == 2);
  REQUIRE(topK[0]->id == "id3");
  REQUIRE(topK[1]->id == "id2");
}