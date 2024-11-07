/*
 * Copyright (C) 2024 by the INTELLI team
 * Created on: 24-11-6 下午3:28
 * Description: ${DESCRIPTION}
 */

#include <torch/torch.h>
#include <Algorithms/LSH/LSHSearch.hpp>
#include <catch2/catch_test_macros.hpp>

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


    torch::Tensor SearchQuery = Data.slice(0, 0, 2);
    std::cout << "The tensors to be queried (first 2 rows): " << SearchQuery << std::endl;

    auto Results = lsh.searchTensor(SearchQuery, 5);
    torch::Tensor result;

    for (int m = 0; m < Results.size(); ++m) {
      result = Results[m];
      std::cout << "Row[" << m <<"] The most similar 5 tensors before deletion:" << std::endl;
      for (int j = 0; j < result.size(0); ++j) {
        int64_t id = result[j][0].item<int64_t>();
        torch::Tensor row = Data[id].unsqueeze(0);
        std::cout << "id " << id << ": " << row << std::endl;
      }
      std::cout << std::endl;
    }
    REQUIRE(Results.size() == 2);
    REQUIRE(Results[0].size(0) == 5);


    REQUIRE(lsh.deleteTensor(SearchQuery, 2));


    Results = lsh.searchTensor(SearchQuery, 5);

    for (int m = 0; m < Results.size(); ++m) {
      result = Results[m];
      std::cout << "Row[" << m <<"] The most similar 5 tensors after removing:" << std::endl;
      for (int j = 0; j < result.size(0); ++j) {
        int64_t id = result[j][0].item<int64_t>();
        torch::Tensor row = Data[id].unsqueeze(0);
        std::cout << "id " << id << ": " << row << std::endl;
      }
      std::cout << std::endl;
    }

    REQUIRE(Results.size() == 2);
    REQUIRE(Results[0].size(0) == 5);
  }


  SECTION("Revise Tensor") {
    LshSearch lsh(Dimensions, NumPlanes);
    torch::Tensor Data = torch::randn({20000, Dimensions});
    REQUIRE(lsh.insertTensor(Data));


    torch::Tensor reviseQuery = Data.slice(0, 0, 3);
    torch::Tensor NewTensor = torch::randn({3, Dimensions});

    std::cout << "Primitive tensors: " << reviseQuery << std::endl;
    std::cout << "New Tensors: " << NewTensor << std::endl;

    REQUIRE(lsh.reviseTensor(reviseQuery, NewTensor));

    Data.slice(0, 0, 3) = NewTensor;

    auto Results = lsh.searchTensor(NewTensor, 5);
    for (int m = 0; m<Results.size(); ++m) {
      auto result = Results[m];
      std::cout << "Row[" << m<< "] Query after modifying the tensors" << std::endl;
      for (int j = 0; j < result.size(0); ++j) {
        int64_t id = result[j][0].item<int64_t>();
        torch::Tensor row = Data[id].unsqueeze(0);
        std::cout << "id " << id << ": " << row << std::endl;
      }
      std::cout << std::endl;
    }

    REQUIRE(Results.size() == 3);
    REQUIRE(Results[0].size(0) == 5);
  }
}
