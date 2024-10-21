/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/9
 * Description: [Provide description here]
 */
#ifndef INTELLISTREAM_SRC_CORE_VECTOR_DB_HPP_
#define INTELLISTREAM_SRC_CORE_VECTOR_DB_HPP_

#ifndef VECTOR_DB_H
#define VECTOR_DB_H

#include <Algorithms/search_algorithm.hpp> // Interface for search algorithms (e.g., k-NN, ANNS)
#include <unordered_map>
#include <vector>
#include <shared_mutex>  // For shared_mutex, unique_lock, shared_lock
#include <mutex>         // For std::mutex
#include <queue>
#include <thread>
#include <functional>
#include <memory>

class VectorDB {
 public:
  // Constructor and Destructor
  VectorDB(size_t dimensions, std::shared_ptr<SearchAlgorithm> search_algorithm = nullptr);
  ~VectorDB();

  // Insert a vector into the database (either from streaming or directly)
  bool insert_vector(const std::vector<float>& vec);

  bool update_vector(size_t id, const std::vector<float> &new_vec);

  bool remove_vector(size_t id);

  // Query the nearest vectors using a k-NN search or other search algorithms
  std::vector<std::vector<float>> query_nearest_vectors(const std::vector<float>& query_vec, size_t k) const;

  // Stream support: Insert vectors continuously and process them in parallel
  void start_streaming();
  void stop_streaming();

  void insert_streaming_vector(const std::vector<float>& vec);  // Insert vector into the stream buffer

  // Streaming helper functions
  void process_streaming_queue();  // Process tuples from the streaming queue
  int get_dimensions();

 private:
  // Internal storage for vectors (indexed by ID)
  std::unordered_map<size_t, std::vector<float>> vector_store;

  // Search algorithm for querying (e.g., k-NN, Approximate NN)
  std::shared_ptr<SearchAlgorithm> search_algorithm;

  // Thread-safe data structures for concurrency
  mutable std::shared_mutex db_mutex;
  std::queue<std::vector<float>> stream_buffer;//to hold input stream
  size_t buffer_size;
  bool is_running;

  // Parallel processing: thread workers
  std::vector<std::thread> workers;

  // Dimensions of the vectors
  size_t dimensions;

  // ID generation for vectors
  size_t next_id = 0;
  size_t generate_id();

  // Helper function for thread safety
  void start_workers();
  void stop_workers();
};


#endif // VECTOR_DB_H

#endif //INTELLISTREAM_SRC_CORE_VECTOR_DB_HPP_
