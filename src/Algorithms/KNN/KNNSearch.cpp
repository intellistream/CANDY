/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/9
 * Description: [Provide description here]
 */
#include <Algorithms/KNN/KNNSearch.hpp>
#include <algorithm>

// Constructor with vector dimensions
KnnSearch::KnnSearch(size_t dimensions) : dimensions(dimensions) {
}

// Reset the current index
void KnnSearch::reset() {
    index.clear();
}

// Set configuration (placeholder implementation)
bool KnnSearch::setConfig(INTELLI::ConfigMapPtr cfg) {
    // Implement the logic to handle configurations if needed
    return true;
}

// Start High-Performance Computation (HPC) placeholder implementation
bool KnnSearch::startHPC() {
    // Implement the logic to handle starting HPC if needed
    return true;
}

// End High-Performance Computation (HPC) placeholder implementation
bool KnnSearch::endHPC() {
    // Implement the logic to handle ending HPC if needed
    return true;
}


// Insert tensor into the index
bool KnnSearch::insertTensor(torch::Tensor &t) {
    if (t.size(1) != dimensions) {
        return false; // Dimension mismatch
    }

    for (int64_t i = 0; i < t.size(0); ++i) {
        auto tensor_row = t[i];
        std::vector<float> vec(tensor_row.data_ptr<float>(), tensor_row.data_ptr<float>() + dimensions);
        index[i] = vec;
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

// Revise tensor in the index
bool KnnSearch::reviseTensor(torch::Tensor &t, torch::Tensor &w) {
    if (t.size(1) != dimensions || w.size(1) != dimensions) {
        return false; // Dimension mismatch
    }

    for (int64_t i = 0; i < t.size(0); ++i) {
        auto tensor_row = w[i];
        std::vector<float> vec(tensor_row.data_ptr<float>(), tensor_row.data_ptr<float>() + dimensions);
        index[i] = vec;
    }
    return true;
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

// Helper function to calculate Euclidean distance
float KnnSearch::calculate_distance(const std::vector<float> &vec1, const std::vector<float> &vec2) const {
    float sum = 0.0f;
    for (size_t i = 0; i < dimensions; ++i) {
        float diff = vec1[i] - vec2[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// Example of finding the k-nearest neighbors
std::vector<size_t> KnnSearch::findKNearestNeighbors(const std::vector<float> &query, size_t k) const {
    std::vector<std::pair<float, size_t> > distances;

    for (const auto &[key, value]: index) {
        float distance = calculate_distance(query, value);
        distances.emplace_back(distance, key);
    }

    std::sort(distances.begin(), distances.end());

    std::vector<size_t> neighbors;
    for (size_t i = 0; i < k && i < distances.size(); ++i) {
        neighbors.push_back(distances[i].second);
    }

    return neighbors;
}
