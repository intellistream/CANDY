/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/9
 * Description: [Provide description here]
 */
#include <Core/vector_db.hpp>
#include <Algorithms/knn_search.hpp>
#include <catch2/catch_test_macros.hpp>
#include <thread>
#include <chrono>

// Mock search algorithm class for testing purposes
#include <cmath>       // For std::sqrt
#include <algorithm>   // For std::sort
#include <map>         // For std::map

class MockSearchAlgorithm : public SearchAlgorithm {
 public:
  MockSearchAlgorithm() = default;  // Default constructor

  void insert(size_t id, const std::vector<float>& vec) override {
    index[id] = vec;
  }

  std::vector<size_t> query(const std::vector<float>& query_vec, size_t k) const override {
    // Vector to store pairs of id and computed distance
    std::vector<std::pair<size_t, float>> id_distance_pairs;

    // Compute the distance between the query vector and each vector in the index
    for (const auto& entry : index) {
      float distance = computeDistance(query_vec, entry.second);
      id_distance_pairs.emplace_back(entry.first, distance);
    }

    // Sort the pairs based on distance (ascending order)
    std::sort(id_distance_pairs.begin(), id_distance_pairs.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    // Collect the IDs of the k nearest vectors
    std::vector<size_t> result_ids;
    for (size_t i = 0; i < std::min(k, id_distance_pairs.size()); ++i) {
      result_ids.push_back(id_distance_pairs[i].first);
    }

    return result_ids;
  }

  void remove(size_t id) override {
    index.erase(id);
  }

 private:
  std::map<size_t, std::vector<float>> index;  // Use std::map for ordered iteration

  // Helper function to compute Euclidean distance between two vectors
  float computeDistance(const std::vector<float>& vec1, const std::vector<float>& vec2) const {
    if (vec1.size() != vec2.size()) {
      // Handle dimension mismatch if necessary
      return std::numeric_limits<float>::max();
    }
    float sum = 0.0f;
    for (size_t i = 0; i < vec1.size(); ++i) {
      float diff = vec1[i] - vec2[i];
      sum += diff * diff;
    }
    return std::sqrt(sum);
  }
};

TEST_CASE("VectorDB: Insert and Query Operations") {
  size_t dimensions = 3;
  auto search_algorithm = std::make_shared<MockSearchAlgorithm>();
  VectorDB db(dimensions, search_algorithm);

  SECTION("Insert a vector with correct dimensions") {
    std::vector<float> vec = {1.0, 2.0, 3.0};
    REQUIRE(db.insert_vector(vec) == true);
  }

  SECTION("Insert a vector with incorrect dimensions") {
    std::vector<float> vec = {1.0, 2.0};
    REQUIRE(db.insert_vector(vec) == false);
  }

  SECTION("Query nearest vectors after inserting vectors") {
    std::vector<float> vec1 = {1.0, 2.0, 3.0};
    std::vector<float> vec2 = {4.0, 5.0, 6.0};
    db.insert_vector(vec1);
    db.insert_vector(vec2);

    std::vector<float> query_vec = {1.0, 2.0, 3.0};
    auto results = db.query_nearest_vectors(query_vec, 2);
    REQUIRE(results.size() == 2);
  }
}

TEST_CASE("VectorDB: Streaming Operations") {
  size_t dimensions = 3;
  auto search_algorithm = std::make_shared<MockSearchAlgorithm>();
  VectorDB db(dimensions, search_algorithm);

  SECTION("Start and stop streaming engine") {
    db.start_streaming();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    db.stop_streaming();
    REQUIRE(true);  // If no exceptions occur, the test passes
  }

  SECTION("Insert vectors into streaming queue") {
    db.start_streaming();

    std::vector<float> vec1 = {1.0, 2.0, 3.0};
    std::vector<float> vec2 = {4.0, 5.0, 6.0};
    db.insert_streaming_vector(vec1);
    db.insert_streaming_vector(vec2);

    std::this_thread::sleep_for(std::chrono::milliseconds(100));  // Allow some time for processing
    db.stop_streaming();

    std::vector<float> query_vec = {1.0, 2.0, 3.0};
    auto results = db.query_nearest_vectors(query_vec, 2);
    REQUIRE(results.size() == 2);
  }
}

TEST_CASE("VectorDB: Thread Safety") {
  size_t dimensions = 3;
  auto search_algorithm = std::make_shared<MockSearchAlgorithm>();
  VectorDB db(dimensions, search_algorithm);

  SECTION("Concurrent insertion into the database") {
    db.start_streaming();

    auto insert_task = [&db](const std::vector<float> &vec) {
      db.insert_streaming_vector(vec);
    };

    std::vector<float> vec1 = {1.0, 2.0, 3.0};
    std::vector<float> vec2 = {4.0, 5.0, 6.0};
    std::vector<float> vec3 = {7.0, 8.0, 9.0};

    std::thread t1(insert_task, vec1);
    std::thread t2(insert_task, vec2);
    std::thread t3(insert_task, vec3);

    t1.join();
    t2.join();
    t3.join();

    std::this_thread::sleep_for(std::chrono::milliseconds(100));  // Allow some time for processing
    db.stop_streaming();

    std::vector<float> query_vec = {1.0, 2.0, 3.0};
    auto results = db.query_nearest_vectors(query_vec, 3);
    REQUIRE(results.size() == 3);
  }
}

TEST_CASE("VectorDB: Querying Edge Cases") {
  size_t dimensions = 3;
  auto search_algorithm = std::make_shared<MockSearchAlgorithm>();
  VectorDB db(dimensions, search_algorithm);

  SECTION("Query nearest vectors when database is empty") {
    std::vector<float> query_vec = {1.0, 2.0, 3.0};
    auto results = db.query_nearest_vectors(query_vec, 2);
    REQUIRE(results.size() == 0);
  }

  SECTION("Query more vectors than available in the database") {
    std::vector<float> vec1 = {1.0, 2.0, 3.0};
    db.insert_vector(vec1);

    std::vector<float> query_vec = {1.0, 2.0, 3.0};
    auto results = db.query_nearest_vectors(query_vec, 5);
    REQUIRE(results.size() == 1);  // Only one vector available in the database
  }
}