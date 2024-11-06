/*
 * Copyright (C) 2024 by the INTELLI team
 * Created on: 24-11-6 下午3:28
 * Description: ${DESCRIPTION}
 */

#include <catch2/catch_test_macros.hpp>
#include <torch/torch.h>
#include <Algorithms/LSH/LSHSearch.hpp>

using namespace CANDY_ALGO;


TEST_CASE("LSH Search Tests") {
  // Set the dimensions and number of hyperplanes
  size_t Dimensions = 10;
  size_t NumPlanes = 10;

  // Insert/Search/Delete test
  SECTION("Insert/Search/Delete Tensor") {
    LshSearch lsh(Dimensions, NumPlanes);
    torch::Tensor Data = torch::randn({20000, Dimensions});
    REQUIRE(lsh.insertTensor(Data));

    auto SearchQuery = Data[0].unsqueeze(0);
    std::cout << "The tensor to be queried: " << SearchQuery << std::endl;

    // std::cout << "SearchQuery.size(0): " << SearchQuery.size(0) << ", SearchQuery.size(1): " << SearchQuery.size(1) << std::endl;

    auto Results = lsh.searchTensor(SearchQuery, 5);
    torch::Tensor result = Results[0];

    std::cout << "The most similar 5 tensors queried:" << std::endl;
    for (int j = 0; j < result.size(0); ++j) {
      int64_t id = result[j][0].item<int64_t>();
      torch::Tensor row = Data[id].unsqueeze(0);
      std::cout << "id " << id << ": " << row << std::endl;
    }
    std::cout << std::endl;

    REQUIRE(Results.size() == 1);
    REQUIRE(Results[0].size(0) == 5);

    REQUIRE(lsh.deleteTensor(SearchQuery, 1));

    Results = lsh.searchTensor(SearchQuery, 5);
    result = Results[0];

    std::cout << "Query after removing the most similar tensors:" << std::endl;
    for (int j = 0; j < result.size(0); ++j) {
      int64_t id = result[j][0].item<int64_t>();
      torch::Tensor row = Data[id].unsqueeze(0);
      std::cout << "id " << id << ": " << row << std::endl;
    }
    std::cout << std::endl;

    REQUIRE(Results.size() == 1);
    REQUIRE(Results[0].size(0) == 5);
  }

  // Revise test
  SECTION("Revise Tensor") {
    LshSearch lsh(Dimensions, NumPlanes);
    torch::Tensor Data = torch::randn({20000, Dimensions});
    REQUIRE(lsh.insertTensor(Data));

    auto reviseQuery = Data[0].unsqueeze(0);
    torch::Tensor NewTensor = torch::randn({1, Dimensions});
    // Data[0] = NewTensor.squeeze(0);

    std::cout << "Primitive tensors: " << reviseQuery << std::endl;
    std::cout << "New Tensors: " << NewTensor << " " << std::endl;
    REQUIRE(lsh.reviseTensor(reviseQuery, NewTensor));

    Data[0] = NewTensor.squeeze(0);
    auto Results = lsh.searchTensor(NewTensor, 5);
    auto result = Results[0];
    std::cout << "Query after modifying the tensor" << std::endl;
    for (int j = 0; j < result.size(0); ++j) {
      int64_t id = result[j][0].item<int64_t>();
      torch::Tensor row = Data[id].unsqueeze(0);
      std::cout << "id " << id << ": " << row << std::endl;
    }

    REQUIRE(Results.size() == 1);
    REQUIRE(Results[0].size(0) == 5);
  }

}
