/*
 * Copyright (C) 2024 by the INTELLI team
 * Created on: 2024/10/9
 * Description: [Provide description here]
 */
#include <Algorithms/KNN/KNNSearch.hpp>
#include <algorithm>

#include <Utils/Computation.hpp>

// Constructor with vector dimensions
KnnSearch::KnnSearch(size_t dimensions) : dimensions(dimensions) {
}

// Reset the current index
void KnnSearch::reset() {
    index.clear();
}

// Insert tensor into the index
bool KnnSearch::insertTensor(const torch::Tensor &t) {
    if (t.size(1) != dimensions) { // Dimension out of range (expected to be in range of [-1, 0], but got 1)
        return false;
    }

    for (int64_t i = 0; i < t.size(0); ++i) {
        index[i] = t[i]; // Directly assign the row tensor to the index map
    }

    return true;
}


// Load initial tensor into the index
bool KnnSearch::loadInitialTensor(torch::Tensor &t) {
    return insertTensor(t);
}

// Delete tensor from the index
bool KnnSearch::deleteTensor(torch::Tensor &t, int64_t k) {
    if (t.size(1) != dimensions) {
        return false; // Dimension mismatch
    }

    for (int64_t i = 0; i < t.size(0); ++i) {
        index.erase(i);
    }
    return true;
}


bool KnnSearch::reviseTensor(torch::Tensor &t, torch::Tensor &w) {
    if (t.size(1) != dimensions || w.size(1) != dimensions) {
        return false; // Dimension mismatch
    }
    // Assuming each row in `w` corresponds to an entry in `index`
    for (int64_t i = 0; i < t.size(0); ++i) {
        index[i] = w[i]; // Directly assign the row tensor to the index map
    }
    return true;
}

std::vector<torch::Tensor> KnnSearch::searchTensor(const torch::Tensor &q, int64_t k) {
    // Ensure the query tensor has the correct dimensions
    if (q.size(0) != dimensions) {
        std::cerr << "Error: Query tensor dimensions do not match the expected size (" << dimensions << ")." <<
                std::endl;
        return {};
    }

    // Vector to store pairs of distance and corresponding tensor ID
    std::vector<std::pair<float, size_t> > distances;

    // Calculate the Euclidean distance between the query tensor and each tensor in the index
    for (const auto &[id, tensor]: index) {
        float distance = CANDY::computeL2Distance(q.data_ptr<float>(), tensor.data_ptr<float>(), dimensions);
        distances.emplace_back(distance, id);
    }

    // Sort the distances in ascending order
    std::sort(distances.begin(), distances.end(), [](const auto &a, const auto &b) {
        return a.first < b.first;
    });

    // Prepare the output vector for the k-nearest neighbors
    std::vector<torch::Tensor> nearest_neighbors;
    for (int64_t i = 0; i < k && i < distances.size(); ++i) {
        nearest_neighbors.push_back(index[distances[i].second]);
    }

    return nearest_neighbors;
}


// Reset index statistics (placeholder implementation)
bool KnnSearch::resetIndexStatistics() {
    // Implement resetting index statistics if needed
    return true;
}

// Get index statistics (placeholder implementation)
INTELLI::ConfigMapPtr KnnSearch::getIndexStatistics() {
    // Implement retrieving index statistics if needed
    return nullptr;
}


