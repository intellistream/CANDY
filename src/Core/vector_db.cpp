/*
 * Copyright (C) 2024 by the INTELLI team
 * Created on: 2024/10/9
 * Description: [Provide description here]
 */
#include <Algorithms/KNN/KNNSearch.hpp>
#include <Core/vector_db.hpp>
#include <iostream>
#include <thread>
#include <vector>

// Constructor: Initialize the tensor database with a number of dimensions and a search algorithm

VectorDB::VectorDB(size_t dimensions, CANDY_ALGO::ANNSBasePtr ann_algorithm)
    : ann_algorithm(ann_algorithm), is_running(false), dimensions(dimensions) {
  if (!this->ann_algorithm) {
    // Instantiate a default ANNS algorithm if none provided
    this -> ann_algorithm = std::make_shared<CANDY_ALGO::KnnSearch>(dimensions);
    this -> ann_algorithm -> setConfig(nullptr);
  }
}

// Destructor
VectorDB::~VectorDB() {
  stop_streaming();
}

// Generate a new unique ID for each tensor
size_t VectorDB::generate_id() {
  return next_id++;
}

// Insert a tensor directly into the tensor database (exclusive write access)
bool VectorDB::insert_tensor(const torch::Tensor& tensor) {
  {
    std::unique_lock<std::shared_mutex> lock(db_mutex);  // Exclusive write lock
    auto insert_container = torch::zeros({1, tensor.size(0)});
    insert_container[0] = tensor;
    if (ann_algorithm) {
      ann_algorithm->insertTensor(
          insert_container);  // Insert into the ANNS algorithm's index
    }
  }

  return true;
}

// Update an existing tensor (exclusive write access)
bool VectorDB::update_tensor(const torch::Tensor& old_tensor,
                             torch::Tensor& new_tensor) {
  {
    std::unique_lock<std::shared_mutex> lock(db_mutex);  // Exclusive write lock
    auto old_tensor_container = torch::zeros({1, old_tensor.size(0)});
    old_tensor_container[0] = old_tensor;
    auto new_tensor_container = torch::zeros({1, new_tensor.size(0)});
    new_tensor_container[0] = new_tensor;
    if (ann_algorithm) {
      ann_algorithm->reviseTensor(
          old_tensor_container,
          new_tensor_container);  // Revise the tensor in the ANNS index
    }
  }
  return true;
}

bool VectorDB::remove_tensor(const torch::Tensor& tensor) {
  {
    std::unique_lock<std::shared_mutex> lock(db_mutex);  // Exclusive write lock
    auto delete_container = torch::zeros({1, tensor.size(0)});
    delete_container[0] = tensor;
    if (ann_algorithm) {
      ann_algorithm->deleteTensor(delete_container,
                                  1);  // Delete the tensor from the ANNS index
    }
  }

  return true;
}

// Query the nearest tensors using the ANNS algorithm (shared read access)
std::vector<torch::Tensor> VectorDB::query_nearest_tensors(
    const torch::Tensor& query_tensor, size_t k) const {
  std::shared_lock<std::shared_mutex> lock(db_mutex);  // Shared read lock
  if (!ann_algorithm) {
    INTELLI_ERROR("Error: No ANNS algorithm available for querying.");
    return {};
  }
  torch::Tensor query_container = torch::zeros({1, query_tensor.size(0)});
  query_container[0] = query_tensor;
  // Query the nearest k tensors directly from the ANNS algorithm
  std::vector<torch::Tensor> nearest_tensors =
      ann_algorithm->searchTensor(query_container, k);
  return nearest_tensors;
}

// Start the streaming engine (processes stream buffers in parallel)
void VectorDB::start_streaming() {
  is_running = true;
  start_workers();  // Start the worker threads
}

// Stop the streaming engine
void VectorDB::stop_streaming() {
  is_running = false;
  stop_workers();  // Stop the worker threads
}

// Insert a tensor into the streaming queue (exclusive write access)
void VectorDB::insert_streaming_tensor(const torch::Tensor& tensor) {
  {
    std::unique_lock<std::shared_mutex> lock(
        db_mutex);               // Exclusive write lock for streaming insert
    stream_buffer.push(tensor);  // Push the tensor into the streaming queue
  }
}

// Process the streaming queue (insert tensors into the database in parallel)
void VectorDB::process_streaming_queue() {
  while (is_running) {
    torch::Tensor tensor;
    {
      std::unique_lock<std::shared_mutex> lock(
          db_mutex);  // Exclusive access to remove from queue
      if (!stream_buffer.empty()) {
        tensor = stream_buffer.front();
        stream_buffer.pop();
      }
    }

    if (tensor.defined()) {
      insert_tensor(tensor);  // Insert the tensor into the database
    }
  }
}

int VectorDB::get_dimensions() const {
  return dimensions;
}

// Start the worker threads to process the streaming queue
void VectorDB::start_workers() {
  for (int i = 0; i < std::thread::hardware_concurrency(); ++i) {
    workers.emplace_back(&VectorDB::process_streaming_queue, this);
  }
}

// Stop the worker threads
void VectorDB::stop_workers() {
  for (auto& worker : workers) {
    if (worker.joinable()) {
      worker.join();  // Ensure all threads are properly joined
    }
  }
  workers.clear();
}
