/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/9
 * Description: [Provide description here]
 */
#include <Core/vector_db.hpp>
#include <iostream>
#include <thread>
#include <Algorithms/knn_search.hpp>

// Constructor: Initialize the vector database with a number of dimensions and a search algorithm
VectorDB::VectorDB(size_t dimensions, std::shared_ptr<SearchAlgorithm> search_algorithm)
  : dimensions(dimensions), search_algorithm(search_algorithm), is_running(false) {
  if (!this->search_algorithm) {
    this->search_algorithm = std::make_shared<KnnSearch>(dimensions);
  }
}

// Destructor
VectorDB::~VectorDB() {
  stop_streaming();
}

// Generate a new unique ID for each vector
size_t VectorDB::generate_id() {
  return next_id++;
}

// Insert a vector directly into the vector database (exclusive write access)
bool VectorDB::insert_vector(const std::vector<float> &vec) {
  if (vec.size() != dimensions) {
    std::cerr << "Error: Vector dimensions do not match the expected size (" << dimensions << ")." << std::endl;
    return false;
  }

  {
    std::unique_lock<std::shared_mutex> lock(db_mutex);  // Exclusive write lock
    size_t id = generate_id();
    vector_store[id] = vec;  // Store the vector in the database
    if (search_algorithm) {
      search_algorithm->insert(id, vec);  // Insert into the search algorithm's index
    }
  }

  return true;
}

// Update an existing vector (exclusive write access)
bool VectorDB::update_vector(size_t id, const std::vector<float> &new_vec) {
  if (new_vec.size() != dimensions) {
    std::cerr << "Error: Vector dimensions do not match the expected size (" << dimensions << ")." << std::endl;
    return false;
  } {
    std::unique_lock<std::shared_mutex> lock(db_mutex); // Exclusive write lock
    if (vector_store.find(id) == vector_store.end()) {
      std::cerr << "Error: Vector with ID " << id << " not found." << std::endl;
      return false;
    }
    vector_store[id] = new_vec; // Update the vector in the database
    if (search_algorithm) {
      // search_algorithm->update(id, new_vec); // Update the vector in the search algorithm's index
    }
  }

  return true;
}

// Delete a vector by its ID (exclusive write access)
bool VectorDB::remove_vector(size_t id) { {
    std::unique_lock<std::shared_mutex> lock(db_mutex); // Exclusive write lock
    auto it = vector_store.find(id);
    if (it == vector_store.end()) {
      std::cerr << "Error: Vector with ID " << id << " not found." << std::endl;
      return false;
    }
    vector_store.erase(it); // Remove the vector from the database
    if (search_algorithm) {
      search_algorithm->remove(id); // Remove the vector from the search algorithm's index
    }
  }

  return true;
}


// Query the nearest vectors using the search algorithm (shared read access)
std::vector<std::vector<float>> VectorDB::query_nearest_vectors(const std::vector<float> &query_vec, size_t k) const {
  if (query_vec.size() != dimensions) {
    std::cerr << "Error: Query vector dimensions do not match the expected size (" << dimensions << ")." << std::endl;
    return {};
  }

  std::shared_lock<std::shared_mutex> lock(db_mutex);  // Shared read lock
  if (!search_algorithm) {
    std::cerr << "Error: No search algorithm available for querying." << std::endl;
    return {};
  }

  // Query the nearest k vectors using the search algorithm
  std::vector<size_t> nearest_ids = search_algorithm->query(query_vec, k);

  // Retrieve the actual vectors based on the IDs returned
  std::vector<std::vector<float>> nearest_vectors;
  for (size_t id : nearest_ids) {
    nearest_vectors.push_back(vector_store.at(id));
  }

  return nearest_vectors;
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

// Insert a vector into the streaming queue (exclusive write access)
void VectorDB::insert_streaming_vector(const std::vector<float> &vec) {
  if (vec.size() != dimensions) {
    std::cerr << "Error: Vector dimensions do not match the expected size (" << dimensions << ")." << std::endl;
    return;
  }

  {
    std::unique_lock<std::shared_mutex> lock(db_mutex);  // Exclusive write lock for streaming insert
    stream_buffer.push(vec);  // Push the vector into the streaming queue
  }
}

// Process the streaming queue (insert vectors into the database in parallel)
void VectorDB::process_streaming_queue() {
  while (is_running) {
    std::vector<float> vec;
    {
      std::unique_lock<std::shared_mutex> lock(db_mutex);  // Exclusive access to remove from queue
      if (!stream_buffer.empty()) {
        vec = stream_buffer.front();
        stream_buffer.pop();
      }
    }

    if (!vec.empty()) {
      insert_vector(vec);  // Insert the vector into the database
    }
  }
}

int VectorDB::get_dimensions() {
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
  for (auto &worker : workers) {
    if (worker.joinable()) {
      worker.join();  // Ensure all threads are properly joined
    }
  }
  workers.clear();
}