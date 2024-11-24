/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: jjzhao
 * Created on: 2024/11/24
 * Description: [Provide description here]
 */
#ifndef INTELLISTREAM_SRC_CORE_VECTOR_DATABASE_HPP_
#define IINTELLISTREAM_SRC_CORE_VECTOR_DATABASE_HPP_

#ifndef VECTOR_DATABASE_H
#define VECTOR_DATABASE_H

#include <Algorithms/SeparateANNSBase.hpp>
#include <torch/torch.h>
#include <string>
#include <thread>
#include <vector>

class VectorDatabase {
 public:
  // Constructor and Destructor

  VectorDatabase();

  ~VectorDatabase();

  bool insert_tensor_rawid (const torch::Tensor&tensor, int rawId);

  std::string displayStore();

  std::vector<int> delete_tensor(const torch::Tensor& tensor, size_t k);

  std::vector<int> search_tensor(const torch::Tensor& query_tensor, size_t k);

 private:
  // ANNS algorithm for querying (e.g., k-NN, Approximate NN)
  SeparateANNSBase anns_algorithm;

  // Parallel processing: thread workers
  std::vector<std::thread> workers;

  // Helper function for thread safety
  void start_workers();

  void stop_workers();
};

#endif  // VECTOR_DB_H

#endif  //INTELLISTREAM_SRC_CORE_VECTOR_DB_HPP_
