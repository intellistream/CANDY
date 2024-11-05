/*
 * Copyright (C) 2024 by the INTELLI team
 * Created on: 2024/10/10
 * Description: [Provide description here]
 */
#include <API/StreamAPI.hpp>
#include <Core/vector_db.hpp>

// Constructor
VectorDBStream::VectorDBStream() {}

// Stream definition from input tensors
VectorDBStream VectorDBStream::from(
    const std::vector<torch::Tensor>& input_tensors) {
  VectorDBStream stream;
  stream.tensors = input_tensors;
  return stream;
}

// Map transformation: Applies a function to each tensor
VectorDBStream& VectorDBStream::map(
    const std::function<torch::Tensor(const torch::Tensor&)>& func) {
  for (auto& tensor : tensors) {
    tensor = func(tensor);  // Apply the transformation function
  }
  return *this;
}

// Filter transformation: Filters tensors based on a predicate
VectorDBStream& VectorDBStream::filter(
    const std::function<bool(const torch::Tensor&)>& predicate) {
  std::vector<torch::Tensor> filtered_tensors;
  for (const auto& tensor : tensors) {
    if (predicate(tensor)) {
      filtered_tensors.push_back(tensor);
    }
  }
  tensors = filtered_tensors;
  return *this;
}

// LLM embedding generation: Convert input text to embeddings
VectorDBStream& VectorDBStream::map_to_embedding(
    const std::function<torch::Tensor(const std::string&)>& embedding_func) {
  std::vector<torch::Tensor> embeddings;
  for (const auto& tensor : tensors) {
    std::string text =
        "example text";  // Replace with actual text extraction logic
    embeddings.push_back(embedding_func(text));
  }
  tensors = embeddings;
  return *this;
}

// Sink: Automatically write the results to the vector database for persistent storage
void VectorDBStream::to_sink(const std::shared_ptr<VectorDB>& vector_db) {
  for (const auto& tensor : tensors) {
    vector_db->insert_tensor(
        tensor);  // Insert each tensor into the vector database
  }
}

// Query: Perform real-time queries on the vector database using tensors
std::vector<torch::Tensor> VectorDBStream::query_nearest(
    const torch::Tensor& query_tensor, size_t k) {
  std::shared_ptr<VectorDB> vector_db =
      std::make_shared<VectorDB>(query_tensor.size(0), nullptr);
  return vector_db->query_nearest_tensors(
      query_tensor, k);  // Query nearest tensors in real-time
}