#pragma once

#include <vector>
#include <unordered_map>
#include <queue>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

using Vector = std::vector<float>;

class HNSW {
public:
    explicit HNSW(int dimensions, int maxConnections = 16)
        : dimensions(dimensions), maxConnections(maxConnections), rng(std::random_device{}()) {}

    // Add a vector to the index
    void addVector(int id, const Vector &vec) {
        if (vec.size() != dimensions) {
            throw std::runtime_error("Vector dimensions do not match.");
        }

        // Add the vector to the index
        nodes[id] = vec;

        // If this is the first node, set it as the entry point
        if (entryPoint == -1) {
            entryPoint = id;
            return;
        }

        // Perform insertion into the graph
        insertNode(id, vec);
    }

    // Perform k-NN search for a query vector
    std::vector<int> knnSearch(const Vector &query, int k) const {
        if (nodes.empty()) {
            throw std::runtime_error("Index is empty.");
        }

        // Priority queue to track neighbors (max-heap based on distance)
        auto comp = [](const std::pair<float, int> &a, const std::pair<float, int> &b) {
            return a.first < b.first;
        };
        std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, decltype(comp)> pq(comp);

        // Explore neighbors starting from the entry point
        searchGraph(query, entryPoint, k, pq);

        // Extract results from the priority queue
        std::vector<int> results;
        while (!pq.empty()) {
            results.push_back(pq.top().second);
            pq.pop();
        }

        std::reverse(results.begin(), results.end());
        return results;
    }

private:
    int dimensions;
    int maxConnections;
    std::unordered_map<int, Vector> nodes;       // ID to vector mapping
    std::unordered_map<int, std::vector<int>> graph; // Adjacency list representation
    int entryPoint = -1;                         // Entry point for search
    mutable std::mt19937 rng;                    // Random number generator

    // Calculate Euclidean distance
    float calculateDistance(const Vector &a, const Vector &b) const {
        float sum = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            sum += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return std::sqrt(sum);
    }

    // Insert a node into the graph
    void insertNode(int id, const Vector &vec) {
        // Find nearest neighbors
        auto neighbors = knnSearch(vec, maxConnections);

        // Connect the new node to the nearest neighbors
        for (int neighbor : neighbors) {
            graph[neighbor].push_back(id);
            graph[id].push_back(neighbor);

            // Keep the number of connections within the maximum limit
            if (graph[neighbor].size() > maxConnections) {
                pruneConnections(neighbor);
            }
        }
    }

    // Prune connections to enforce maxConnections limit
    void pruneConnections(int node) {
        auto &neighbors = graph[node];

        // Sort neighbors by distance
        std::sort(neighbors.begin(), neighbors.end(),
                  [this, &node](int a, int b) {
                      return calculateDistance(nodes[node], nodes[a]) <
                             calculateDistance(nodes[node], nodes[b]);
                  });

        // Keep only the closest maxConnections neighbors
        if (neighbors.size() > maxConnections) {
            neighbors.resize(maxConnections);
        }
    }

    // Search the graph for k-NN
    void searchGraph(const Vector &query, int startNode, int k,
                     std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>> &pq) const {
        std::unordered_map<int, bool> visited;
        std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<>> candidates;

        candidates.push({calculateDistance(query, nodes.at(startNode)), startNode});

        while (!candidates.empty()) {
            auto [dist, current] = candidates.top();
            candidates.pop();

            if (visited[current]) {
                continue;
            }

            visited[current] = true;

            // Add to result set if better than the worst in the result heap
            if (pq.size() < k || dist < pq.top().first) {
                pq.push({dist, current});
                if (pq.size() > k) {
                    pq.pop();
                }
            }

            // Explore neighbors
            for (int neighbor : graph.at(current)) {
                if (!visited[neighbor]) {
                    candidates.push({calculateDistance(query, nodes.at(neighbor)), neighbor});
                }
            }
        }
    }
};
