/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Junyao Dong
 * Created on: 2024/10/17
 * Description: [Provide description here]
 */
#include <Utils/file_loader.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <fstream>
#include <vector>
#include <iostream>
#include <cmath>
#include <cstdio>

using namespace std;

void write_fvecs(const string& filename, const vector<vector<float>>& data) {
  ofstream file(filename, ios::binary);
  if (!file.is_open()) {
    cerr << "Failed to open file for writing: " << filename << endl;
    return;
  }

  for (const auto& vec : data) {
    int dim = static_cast<int>(vec.size());
    file.write(reinterpret_cast<const char*>(&dim), sizeof(dim)); 
    file.write(reinterpret_cast<const char*>(vec.data()), dim * sizeof(float)); 
  }

  file.close();
}

void write_ivecs(const string& filename, const vector<vector<int>>& data) {
  ofstream file(filename, ios::binary);
  if (!file.is_open()) {
    cerr << "Failed to open file for writing: " << filename << endl;
    return;
  }

  for (const auto& vec : data) {
    int dim = static_cast<int>(vec.size());
    file.write(reinterpret_cast<const char*>(&dim), sizeof(dim)); 
    file.write(reinterpret_cast<const char*>(vec.data()), dim * sizeof(int)); 
  }

  file.close();
}

TEST_CASE("Read fvecs file") {
  vector<vector<float>> test_data = {
    {1.0f, 2.0f, 3.0f},    
    {4.0f, 5.0f, 6.0f},   
    {7.0f, 8.0f, 9.0f}     
  };

  string fname = "test.fvecs";
  write_fvecs(fname, test_data);

  float* data_load = NULL;
  unsigned points_num, dim;

  int res = load_fvecs_data(fname, data_load, points_num, dim);
  REQUIRE(res == 0);
  
  REQUIRE(points_num == 3);
  REQUIRE(dim == 3);

  int n = 2;
  vector<float> vec(data_load + n * dim, data_load + (n + 1) * dim);
  for (size_t i = 0; i < 3; ++i) {
    REQUIRE(fabs(test_data[n][i] - vec[i]) < 1e-5);
  }

  remove(fname.c_str());
}

TEST_CASE("Read ivecs file") {
  vector<vector<int>> test_data = {
    {1, 2, 3},    
    {4, 5, 6},   
    {7, 8, 9}     
  };

  string fname = "test.ivecs";
  write_ivecs(fname, test_data);

  int* data_load = NULL;
  unsigned points_num, dim;

  int res = load_ivecs_data(fname, data_load, points_num, dim);
  REQUIRE(res == 0);
  
  REQUIRE(points_num == 3);
  REQUIRE(dim == 3);

  int n = 2;
  vector<int> vec(data_load + n * dim, data_load + (n + 1) * dim);
  REQUIRE_THAT(test_data[n], Catch::Matchers::Equals(vec));

  remove(fname.c_str());
}