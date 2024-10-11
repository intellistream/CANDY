/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/9
 * Description: [Provide description here]
 */
#include <API/vectordb_stream.hpp>
#include <numeric>
#include <iostream>

int main() {
  // Define some input vectors (e.g., embeddings)
  std::vector<std::vector<float>> input_vectors = {
      {1.0, 2.0, 3.0},
      {4.0, 5.0, 6.0},
      {7.0, 8.0, 9.0}
  };

  // Create the vector database for storage and querying
  std::shared_ptr<VectorDB> vector_db = std::make_shared<VectorDB>(128, nullptr);

  // Flink-style API for vector streaming, transformation, and storage
  VectorDBStream::from(input_vectors)
      .map([](const std::vector<float>& vec) {
        // Example map transformation: multiply each element by 2
        std::vector<float> transformed_vec(vec.size());
        std::transform(vec.begin(), vec.end(), transformed_vec.begin(), [](float v) { return v * 2; });
        return transformed_vec;
      })
      .filter([](const std::vector<float>& vec) {
        // Example filter: keep vectors with a sum of elements greater than 10
        return std::accumulate(vec.begin(), vec.end(), 0.0) > 10.0;
      })
      .map_to_embedding([](const std::string& text) {
        // Example LLM embedding generation (placeholder logic)
        std::vector<float> embedding = {1.0, 2.0, 3.0};  // Replace with real embedding from LLM
        return embedding;
      })
      .to_sink(vector_db);  // Store all the final vectors in the vector database

  // Example real-time query
  std::vector<float> query_vec = {1.0, 2.0, 3.0};  // Example query vector
  std::vector<std::vector<float>> results = VectorDBStream::from({}).query_nearest(query_vec, 3);

  // Print the results
  for (const auto& result_vec : results) {
    for (float val : result_vec) {
      std::cout << val << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}
