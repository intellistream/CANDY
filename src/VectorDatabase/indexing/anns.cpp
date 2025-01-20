#include <VectorDatabase/indexing/anns.hpp>

// Add a vector to the index
void ANNS::addVector(int id, const Vector &vec) {
  if (vec.size() != dimensions) {
    throw std::invalid_argument("Vector dimensions do not match.");
  }
  index[id] = vec;
}

// Perform a k-NN search for a query vector
std::vector<int> ANNS::knnSearch(const Vector &query, int k) const {
  if (query.size() != dimensions) {
    throw std::invalid_argument("Query vector dimensions do not match.");
  }

  // Min-heap to store top-k results (distance, vector ID)
  auto comp = [](const std::pair<float, int> &a, const std::pair<float, int> &b) {
    return a.first < b.first; // Max-heap based on distance
  };
  std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, decltype(comp)> pq(comp);

  for (const auto &[id, vec] : index) {
    float distance = calculateDistance(query, vec);
    pq.push({distance, id});
    if (pq.size() > k) {
      pq.pop(); // Remove the farthest vector
    }
  }

  // Extract results from the heap
  std::vector<int> result;
  while (!pq.empty()) {
    result.push_back(pq.top().second);
    pq.pop();
  }

  // Reverse the order to have nearest neighbors first
  std::reverse(result.begin(), result.end());
  return result;
}

// Remove a vector from the index
void ANNS::removeVector(int id) {
  if (index.find(id) == index.end()) {
    throw std::invalid_argument("Vector ID not found in the index.");
  }
  index.erase(id);
}

// Calculate Euclidean distance between two vectors
float ANNS::calculateDistance(const Vector &a, const Vector &b) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("Vector dimensions do not match.");
  }

  float sum = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    sum += (a[i] - b[i]) * (a[i] - b[i]);
  }
  return std::sqrt(sum);
}
