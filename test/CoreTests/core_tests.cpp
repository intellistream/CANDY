/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/9
 * Description: [Provide description here]
 */

#include <iostream>
#include <Core/vector_db.hpp>
#include <Algorithms/knn_search.hpp>

void test_basic_insertion_and_retrieval() {
  std::cout << "Running basic insertion and retrieval test..." << std::endl;

  // Use k-NN search algorithm
  std::shared_ptr<SearchAlgorithm> knn_search = std::make_shared<KnnSearch>(128);

  // Create the vector database with 128 dimensions and k-NN search algorithm
  VectorDB db(128, knn_search);

  // Create a vector and insert it into the database
  std::vector<float> vec1(128, 1.0f);  // A vector of 128 dimensions, all set to 1.0
  db.insert_vector(vec1);

  // Query the same vector
  std::vector<std::vector<float>> results = db.query_nearest_vectors(vec1, 1);

  if (results.size() == 1 && results[0] == vec1) {
    std::cout << "Basic insertion and retrieval test passed!" << std::endl;
  } else {
    std::cout << "Basic insertion and retrieval test failed!" << std::endl;
  }
}

int main() {
  test_basic_insertion_and_retrieval();
  return 0;
}
