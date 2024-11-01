/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/9
 * Description: [Provide description here]
 */
#include <Algorithms/knn_search.hpp>  // Header for k-NN search algorithm
#include <Core/vector_db.hpp>  // Header for core vector database operations
#include <iostream>
#include <memory>  // For shared_ptr
#include <string>
#include <vector>

// Function to display a vector result
void display_result(const std::vector<float>& result) {
  std::cout << "[";
  for (size_t i = 0; i < result.size(); ++i) {
    std::cout << result[i];
    if (i != result.size() - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;
}

// Function to process k-Nearest Neighbor search query
void process_knn_query(VectorDB& db) {
  std::cout << "Enter query vector (comma separated values): ";
  std::string input;
  std::getline(std::cin, input);

  // Convert input string to vector of floats
  std::vector<float> query_vector;
  size_t pos = 0;
  while ((pos = input.find(",")) != std::string::npos) {
    query_vector.push_back(std::stof(input.substr(0, pos)));
    input.erase(0, pos + 1);
  }
  query_vector.push_back(std::stof(input));

  std::cout << "Enter value of k: ";
  size_t k;
  std::cin >> k;
  std::cin.ignore();

  // Perform k-NN search using the core vector database
  std::vector<std::vector<float>> results =
      db.query_nearest_vectors(query_vector, k);

  std::cout << "Top-" << k << " results (IDs): " << std::endl;
  for (const auto& result : results) {
    display_result(result);
  }
}

// Function to process Approximate Nearest Neighbor search query
void process_ann_query(VectorDB& db) {
  std::cout << "Enter query vector (comma separated values): ";
  std::string input;
  std::getline(std::cin, input);

  // Convert input string to vector of floats
  std::vector<float> query_vector;
  size_t pos = 0;
  while ((pos = input.find(",")) != std::string::npos) {
    query_vector.push_back(std::stof(input.substr(0, pos)));
    input.erase(0, pos + 1);
  }
  query_vector.push_back(std::stof(input));

  std::cout << "Enter value of k: ";
  size_t k;
  std::cin >> k;
  std::cin.ignore();

  // Perform ANN search using the core vector database
  std::vector<std::vector<float>> results =
      db.query_nearest_vectors(query_vector, k);

  std::cout << "Top-" << k << " approximate results (IDs): " << std::endl;
  for (const auto& result : results) {
    display_result(result);
  }
}

int main() {
  // Initialize VectorDB object with a k-NN search algorithm
  size_t dimensions = 3;  // Assuming vectors have 3 dimensions for this example
  std::shared_ptr<SearchAlgorithm> knn_algorithm =
      std::make_shared<KnnSearch>(dimensions);
  VectorDB db(dimensions, knn_algorithm);
  db.insert_vector({1.0f, 2.0f, 3.0f});  // Example data insertion
  db.insert_vector({4.0f, 5.0f, 6.0f});
  db.insert_vector({7.0f, 8.0f, 9.0f});

  std::cout << "Welcome to the VectorDB Query Tool!" << std::endl;
  bool running = true;

  while (running) {
    std::cout << "\nSelect an option:" << std::endl;
    std::cout << "1. k-Nearest Neighbor (k-NN) Search" << std::endl;
    std::cout << "2. Approximate Nearest Neighbor (ANN) Search" << std::endl;
    std::cout << "3. Exit" << std::endl;
    std::cout << "Enter your choice: ";

    int choice;
    std::cin >> choice;
    std::cin.ignore();  // Ignore newline character left in buffer

    switch (choice) {
      case 1:
        process_knn_query(db);
        break;
      case 2:
        process_ann_query(db);
        break;
      case 3:
        running = false;
        break;
      default:
        std::cout << "Invalid choice. Please try again." << std::endl;
        break;
    }
  }

  std::cout << "Goodbye!" << std::endl;
  return 0;
}