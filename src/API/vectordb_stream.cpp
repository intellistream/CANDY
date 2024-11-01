/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/10
 * Description: [Provide description here]
 */
#include <API/vectordb_stream.hpp>
#include <Core/vector_db.hpp>

// Constructor
VectorDBStream::VectorDBStream() {}

// Stream definition from input vectors
VectorDBStream VectorDBStream::from(
    const std::vector<std::vector<float>>& input_vectors) {
  VectorDBStream stream;
  stream.vectors = input_vectors;
  return stream;
}

// Map transformation: Applies a function to each vector
VectorDBStream& VectorDBStream::map(
    const std::function<std::vector<float>(const std::vector<float>&)>& func) {
  for (auto& vec : vectors) {
    vec = func(vec);  // Apply the transformation function
  }
  return *this;
}

// Filter transformation: Filters vectors based on a predicate
VectorDBStream& VectorDBStream::filter(
    const std::function<bool(const std::vector<float>&)>& predicate) {
  std::vector<std::vector<float>> filtered_vectors;
  for (const auto& vec : vectors) {
    if (predicate(vec)) {
      filtered_vectors.push_back(vec);
    }
  }
  vectors = filtered_vectors;
  return *this;
}

// LLM embedding generation: Convert input text to embeddings
VectorDBStream& VectorDBStream::map_to_embedding(
    const std::function<std::vector<float>(const std::string&)>&
        embedding_func) {
  std::vector<std::vector<float>> embeddings;
  for (const auto& vec : vectors) {
    std::string text =
        "example text";  // Replace with actual text extraction logic
    embeddings.push_back(embedding_func(text));
  }
  vectors = embeddings;
  return *this;
}

// Sink: Automatically write the results to the vector database for persistent storage
void VectorDBStream::to_sink(const std::shared_ptr<VectorDB>& vector_db) {
  for (const auto& vec : vectors) {
    vector_db->insert_vector(
        vec);  // Insert each vector into the vector database
  }
}

// Query: Perform real-time queries on the vector database using embeddings
std::vector<std::vector<float>> VectorDBStream::query_nearest(
    const std::vector<float>& query_vec, size_t k) {
  std::shared_ptr<VectorDB> vector_db =
      std::make_shared<VectorDB>(query_vec.size(), nullptr);
  return vector_db->query_nearest_vectors(
      query_vec, k);  // Query nearest vectors in real-time
}