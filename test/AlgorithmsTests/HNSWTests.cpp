/*
 * Copyright (C) 2024 by the INTELLI team
 * Created on: 24-11-9
 * Description: ${DESCRIPTION}
 */

#include <torch/torch.h>
#include <Algorithms/HNSW/HNSW.hpp>
#include <cassert>
#include <catch2/catch_test_macros.hpp>

using namespace CANDY_ALGO;
const std::string candy_path = CANDY_PATH;

TEST_CASE("HNSW Search Tests") {
  // Set the dimensions and number of hyperplanes
  size_t Dimensions = 10;
  INTELLI::ConfigMapPtr inMap = newConfigMap();
  std::string fileName = candy_path + "/config/configHNSW.csv";
  if (inMap->fromFile(fileName)) {
    INTELLI_INFO("Config loaded from file: " + fileName);
  } else {
    INTELLI_ERROR("Failed to load config from file: " + fileName);
  }
  // Insert/Search/Delete test
  SECTION("Insert/Search/Delete Tensor") {
    HNSW hnsw{};
    hnsw.setConfig(inMap);
    torch::Tensor Data = torch::randn({200, static_cast<long>(Dimensions)});
    REQUIRE(hnsw.insertTensor(Data));
    torch::Tensor SearchQuery = Data.slice(0, 0, 2);
    auto Results = hnsw.searchTensor(SearchQuery, 1);
    torch::Tensor result;
    for (int m = 0; m < Results.size(); ++m) {
      result = Results[m];
      for (int j = 0; j < result.size(0); ++j) {
        int64_t id = result[j].item<int64_t>();
        REQUIRE(id == m);
      }
    }
    REQUIRE(Results.size() == 2);
    REQUIRE(Results[0].size(0) == 1);

    REQUIRE(hnsw.deleteTensor(SearchQuery, 1));

    Results = hnsw.searchTensor(SearchQuery, 1);

    for (int m = 0; m < Results.size(); ++m) {
      result = Results[m];
      for (int j = 0; j < result.size(0); ++j) {
        int64_t id = result[j].item<int64_t>();
        REQUIRE(id != m);
      }
    }

    REQUIRE(Results.size() == 2);
    REQUIRE(Results[0].size(0) == 1);
  }

  SECTION("Revise Tensor") {
    HNSW hnsw{};
    hnsw.setConfig(inMap);
    torch::Tensor Data = torch::randn({200, static_cast<long>(Dimensions)});
    REQUIRE(hnsw.insertTensor(Data));
    torch::Tensor reviseQuery = Data.slice(0, 0, 3);
    torch::Tensor query = reviseQuery.clone();
    torch::Tensor NewTensor = torch::randn({3, static_cast<long>(Dimensions)});
    // promise the reviseQuery is in the hnsw
    auto Results = hnsw.searchTensor(query, 1);
    for (int m = 0; m < Results.size(); ++m) {
      auto result = Results[m];
      for (int j = 0; j < result.size(0); ++j) {
        int64_t id = result[j].item<int64_t>();
        REQUIRE(id == m);
      }
    }
    REQUIRE(hnsw.reviseTensor(reviseQuery, NewTensor));
    Data.slice(0, 0, 3) = NewTensor;
    // now the reviseQuery is should not be in the hnsw
    Results = hnsw.searchTensor(query, 1);
    for (int m = 0; m < Results.size(); ++m) {
      auto result = Results[m];
      for (int j = 0; j < result.size(0); ++j) {
        int64_t id = result[j].item<int64_t>();
        REQUIRE(id != m);
      }
    }
    // try search the new tensor
    Results = hnsw.searchTensor(NewTensor, 1);
    for (int m = 0; m < Results.size(); ++m) {
      auto result = Results[m];
      for (int j = 0; j < result.size(0); ++j) {
        int64_t id = result[j].item<int64_t>();
        REQUIRE(id == m);
      }
    }
    REQUIRE(Results.size() == 3);
    REQUIRE(Results[0].size(0) == 1);
  }
}