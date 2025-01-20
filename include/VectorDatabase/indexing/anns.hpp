#pragma once

#include <vector>
#include <unordered_map>
#include <queue>
#include <cmath>
#include <stdexcept>

using Vector = std::vector<float>;

class ANNS {
public:
  // Constructor
  ANNS(int dimensions) : dimensions(dimensions) {}

  // Add a vector to the index
  void addVector(int id, const Vector &vec);

  // Perform a k-NN search for a query vector
  std::vector<int> knnSearch(const Vector &query, int k) const;

  // Remove a vector from the index
  void removeVector(int id);

private:
  int dimensions; // Dimensionality of vectors
  std::unordered_map<int, Vector> index; // In-memory storage of vectors

  // Calculate Euclidean distance between two vectors
  static float calculateDistance(const Vector &a, const Vector &b);
};
