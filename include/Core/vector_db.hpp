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

#include <Algorithms/SeparateANNSBase.hpp>
#include <memory>
#include <queue>
#include <shared_mutex>  // For shared_mutex, unique_lock, shared_lock
#include <thread>
#include <vector>

class VectorDB {
 public:
  // Constructor and Destructor

  VectorDB(size_t dimensions, CANDY_ALGO::SeparateANNSBasePtr ann_algorithm = nullptr);

  ~VectorDB();

  // Insert, update, and remove tensor-based operations for the database
  bool insert_tensor(const torch::Tensor& tensor);

  bool update_tensor(const torch::Tensor& old_tensor,
                     torch::Tensor& new_tensor);

  bool remove_tensor(const torch::Tensor& tensor);

  // Query using a k-NN search or other specified search algorithms
  std::vector<torch::Tensor> query_nearest_tensors(
      const torch::Tensor& query_tensor, size_t k) const;

  // Stream support: Insert tensors continuously and process them in parallel
  void start_streaming();

  void stop_streaming();

  void insert_streaming_tensor(
      const torch::Tensor& tensor);  // Insert tensor into the stream buffer

  // Streaming helper functions
  void process_streaming_queue();  // Process tensors from the streaming queue
  int get_dimensions() const;

 private:
  // ANNS algorithm for querying (e.g., k-NN, Approximate NN)

  CANDY_ALGO::SeparateANNSBasePtr ann_algorithm;

  // Thread-safe data structures for concurrency
  mutable std::shared_mutex db_mutex;
  std::queue<torch::Tensor> stream_buffer;  // Holds input stream tensors
  size_t buffer_size;
  bool is_running;

  // Parallel processing: thread workers
  std::vector<std::thread> workers;

  // Dimensions of the tensors
  size_t dimensions;

  // ID generation for tensors
  size_t next_id = 0;

  size_t generate_id();

  // Helper function for thread safety
  void start_workers();

  void stop_workers();
};

#endif  // VECTOR_DB_H

#endif  //INTELLISTREAM_SRC_CORE_VECTOR_DB_HPP_
