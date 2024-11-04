/*
 * Copyright (C) 2024 by the INTELLI team
 * Created on: 2024/10/9
 * Description: [Provide description here]
 */
#include <Core/vector_db.hpp>
#include <iostream>
#include <thread>
#include <Algorithms/KNN/KNNSearch.hpp>
// Constructor: Initialize the tensor database with a number of dimensions and a search algorithm

VectorDB::VectorDB(size_t dimensions,CANDY_ALGO::ANNSBasePtr ann_algorithm)
    : dimensions(dimensions), ann_algorithm(ann_algorithm), is_running(false) {
    if (!this->ann_algorithm) {
        // Instantiate a default ANNS algorithm if none provided
        this->ann_algorithm = std::make_shared<CANDY_ALGO::KnnSearch>(dimensions);

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
bool VectorDB::insert_tensor(const torch::Tensor &tensor) {
    if (tensor.size(0) != dimensions) {
        INTELLI_ERROR("Error: Tensor dimensions do not match the expected size (" << dimensions << ").");
        return false;
    } {
        std::unique_lock<std::shared_mutex> lock(db_mutex); // Exclusive write lock
        size_t id = generate_id();
        tensor_store[id] = tensor; // Store the tensor in the database
        if (ann_algorithm) {
            ann_algorithm->insertTensor(tensor); // Insert into the ANNS algorithm's index
        }
    }

    return true;
}

// Update an existing tensor (exclusive write access)
bool VectorDB::update_tensor(size_t id, torch::Tensor &new_tensor) {
    if (new_tensor.size(0) != dimensions) {
        INTELLI_ERROR("Error: Tensor dimensions do not match the expected size (" << dimensions << ").");
        return false;
    } {
        std::unique_lock<std::shared_mutex> lock(db_mutex); // Exclusive write lock
        if (tensor_store.find(id) == tensor_store.end()) {
            INTELLI_ERROR("Error: Tensor with ID " << id << " not found.");
            return false;
        }
        tensor_store[id] = new_tensor; // Update the tensor in the database
        if (ann_algorithm) {
            ann_algorithm->reviseTensor(new_tensor, tensor_store[id]); // Revise the tensor in the ANNS index
        }
    }

    return true;
}

// Delete a tensor by its ID (exclusive write access)
bool VectorDB::remove_tensor(size_t id) { {
        std::unique_lock<std::shared_mutex> lock(db_mutex); // Exclusive write lock
        auto it = tensor_store.find(id);
        if (it == tensor_store.end()) {
            INTELLI_ERROR("Error: Tensor with ID " << id << " not found.");
            return false;
        }
        tensor_store.erase(it); // Remove the tensor from the database
        if (ann_algorithm) {
            ann_algorithm->deleteTensor(it->second, id); // Remove from the ANNS index
        }
    }

    return true;
}

// Query the nearest tensors using the ANNS algorithm (shared read access)
std::vector<torch::Tensor> VectorDB::query_nearest_tensors(const torch::Tensor &query_tensor, size_t k) const {
    if (query_tensor.size(0) != dimensions) {
        INTELLI_ERROR("Error: Query tensor dimensions do not match the expected size (" << dimensions << ").")
        return {};
    }

    std::shared_lock<std::shared_mutex> lock(db_mutex); // Shared read lock
    if (!ann_algorithm) {
        INTELLI_ERROR("Error: No ANNS algorithm available for querying.");
        return {};
    }

    // Query the nearest k tensors directly from the ANNS algorithm
    std::vector<torch::Tensor> nearest_tensors = ann_algorithm->searchTensor(query_tensor, k);

    return nearest_tensors;
}


// Start the streaming engine (processes stream buffers in parallel)
void VectorDB::start_streaming() {
    is_running = true;
    start_workers(); // Start the worker threads
}

// Stop the streaming engine
void VectorDB::stop_streaming() {
    is_running = false;
    stop_workers(); // Stop the worker threads
}

// Insert a tensor into the streaming queue (exclusive write access)
void VectorDB::insert_streaming_tensor(const torch::Tensor &tensor) {
    if (tensor.size(0) != dimensions) {
        INTELLI_ERROR("Error: Tensor dimensions do not match the expected size (" << dimensions << ").");
        return;
    } {
        std::unique_lock<std::shared_mutex> lock(db_mutex); // Exclusive write lock for streaming insert
        stream_buffer.push(tensor); // Push the tensor into the streaming queue
    }
}

// Process the streaming queue (insert tensors into the database in parallel)
void VectorDB::process_streaming_queue() {
    while (is_running) {
        torch::Tensor tensor; {
            std::unique_lock<std::shared_mutex> lock(db_mutex); // Exclusive access to remove from queue
            if (!stream_buffer.empty()) {
                tensor = stream_buffer.front();
                stream_buffer.pop();
            }
        }

        if (tensor.defined()) {
            insert_tensor(tensor); // Insert the tensor into the database
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
    for (auto &worker: workers) {
        if (worker.joinable()) {
            worker.join(); // Ensure all threads are properly joined
        }
    }
    workers.clear();
}
