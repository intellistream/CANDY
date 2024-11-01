/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/10
 * Description: [Provide description here]
 */
#ifndef CANDY_SRC_API_VECTORDB_API_HPP_
#define CANDY_SRC_API_VECTORDB_API_HPP_

#include <Core/vector_db.hpp>
#include <memory>
#include <vector>

#include <functional>
#include <memory>
#include <string>
#include <vector>

class VectorDBStream {
 public:
  // Constructor
  VectorDBStream();

  // Define the stream from input vectors
  static VectorDBStream from(
      const std::vector<std::vector<float>>& input_vectors);

  // Map transformation: Applies a function to each vector
  VectorDBStream& map(
      const std::function<std::vector<float>(const std::vector<float>&)>& func);

  // Filter transformation: Filters vectors based on a predicate
  VectorDBStream& filter(
      const std::function<bool(const std::vector<float>&)>& predicate);

  // LLM embedding generation: Convert input text to embeddings
  VectorDBStream& map_to_embedding(
      const std::function<std::vector<float>(const std::string&)>&
          embedding_func);

  // Sink: Write the results to the vector database (automatically persists results)
  void to_sink(const std::shared_ptr<VectorDB>& vector_db);

  // Real-time query support for streaming RAG systems
  std::vector<std::vector<float>> query_nearest(
      const std::vector<float>& query_vec, size_t k);

 private:
  // Internal vector storage for the streaming data
  std::vector<std::vector<float>> vectors;
};

#endif  //CANDY_SRC_API_VECTORDB_API_HPP_
