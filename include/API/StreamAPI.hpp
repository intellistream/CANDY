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

  // Define the stream from input tensors
  static VectorDBStream from(const std::vector<torch::Tensor>& input_tensors);

  // Map transformation: Applies a function to each tensor
  VectorDBStream& map(
      const std::function<torch::Tensor(const torch::Tensor&)>& func);

  // Filter transformation: Filters tensors based on a predicate
  VectorDBStream& filter(
      const std::function<bool(const torch::Tensor&)>& predicate);

  // LLM embedding generation: Convert input text to embeddings
  VectorDBStream& map_to_embedding(
      const std::function<torch::Tensor(const std::string&)>& embedding_func);

  // Sink: Write the results to the vector database (automatically persists results)
  void to_sink(const std::shared_ptr<VectorDB>& vector_db);

  // Real-time query support for streaming RAG systems
  std::vector<torch::Tensor> query_nearest(const torch::Tensor& query_tensor,
                                           size_t k);

 private:
  // Internal tensor storage for the streaming data
  std::vector<torch::Tensor> tensors;
};

#endif  //CANDY_SRC_API_VECTORDB_API_HPP_
