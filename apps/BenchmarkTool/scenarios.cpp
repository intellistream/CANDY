/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Junyao Dong
 * Created on: 2024/10/16 14:12:04
 * Description: Implementation of various benchmark scenarios for vector database operations.
 */
#include "scenarios.hpp"
#include <Utils/logging.hpp>
#include <Utils/thread_pool.hpp>
#include <Utils/file_loader.hpp>
#include <Utils/ConfigMap.hpp>

#include <vector>
#include <fstream>
#include <random>
#include <future>
#include <filesystem>

using namespace std;

// Available scenarios for benchmarking
const map<string, function<void(VectorDB &db, ScenarioConfig &conf)> > scenarios = {
  {"insert", insert_scenario},
  {"query", query_scenario},
  {"multi_query_insert", multi_query_insert_scenario}
};

// Supported index types
const map<string, string> supported_index = {
  {"hnsw", "[Algorithm-Class-Name]"},
  {"concurrent_hnsw", "[Algorithm-Class-Name]"}
  // Additional index types can be added here
};

// ScenarioConfig constructor implementation
ScenarioConfig::ScenarioConfig(const string &conf_path) {
  load(conf_path);
}

// Loading configuration from the configuration file
void ScenarioConfig::load(const string &conf_path) {
  if (!filesystem::exists(conf_path)) {
    INTELLI_ERROR("Configuration file does not exist: " + conf_path);
    std::cout << "Please double check. The current working directory: " << std::filesystem::current_path() << std::endl;
    throw runtime_error("Configuration file not found: " + conf_path);
  }

  INTELLI::ConfigMapPtr parser = std::make_shared<INTELLI::ConfigMap>();
  parser->parseIni(conf_path);

  scenario_name = parser->getString("scenario_name");
  dataset_path = parser->getString("dataset_path");
  index_type = parser->getString("index_type", "hnsw");
  vector_source = parser->getString("vector_source", "fvecs");
  k_nearest = parser->getInt("k_nearest");
  query_thread_count = parser->getInt("query_thread_count", 1);
  insert_thread_count = parser->getInt("insert_thread_count", 0);
  timeout_in_sec = parser->getInt("timeout_in_sec", 10);
  dimension = parser->getInt("dimension");
}

// Validation check for the configuration
bool ScenarioConfig::isValid() {
  if (scenario_name.empty()) {
    INTELLI_ERROR("Scenario name is empty.");
    return false;
  }
  if (supported_index.find(index_type) == supported_index.end()) {
    INTELLI_ERROR("Index type '" + index_type + "' is not supported.");
    return false;
  }
  if (vector_source.empty()) {
    INTELLI_ERROR("Vector source is empty.");
    return false;
  }
  if (dataset_path.empty()) {
    INTELLI_ERROR("Dataset path is empty.");
    return false;
  }
  if (k_nearest <= 0) {
    INTELLI_ERROR("k_nearest must be greater than 0. Provided value: " + to_string(k_nearest));
    return false;
  }
  return true;
}

// Insert scenario: inserting a set of vectors into the VectorDB
void insert_scenario(VectorDB &db, ScenarioConfig &conf) {
  for (int i = 0; i < 10000; ++i) {
    vector<float> new_vector = {static_cast<float>(i), static_cast<float>(i + 1), static_cast<float>(i + 2)};
    db.insert_vector(new_vector);
  }
}

// Query scenario: querying nearest vectors from the VectorDB
void query_scenario(VectorDB &db, ScenarioConfig &conf) {
  for (int i = 0; i < 10000; ++i) {
    vector<float> query_vector = {static_cast<float>(i), static_cast<float>(i + 1), static_cast<float>(i + 2)};
    db.query_nearest_vectors(query_vector, conf.k_nearest);
  }
}

// Multi-query and insert scenario: performing inserts and queries concurrently
void multi_query_insert_scenario(VectorDB &db, ScenarioConfig &conf) {
  float *data_load = nullptr;
  unsigned points_num, dim;

  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<> dis;

  vector<future<bool> > insert_futures;
  vector<future<vector<vector<float> > > > query_futures;

  ThreadPool pool(conf.query_thread_count + conf.insert_thread_count);
  pool.init();

  // Load data based on vector source type
  if (conf.vector_source == "fvecs") {
    if (load_fvecs_data(conf.dataset_path, data_load, points_num, dim) != 0) {
      INTELLI_ERROR(string("Failed to open fvecs file: ") + conf.dataset_path);
      exit(EXIT_FAILURE);
    }
    dis = uniform_int_distribution<>(0, points_num - 1);
  } else if (conf.vector_source == "hdf5") {
    // if (load_hdf5_data(conf.dataset_path, data_load, points_num, dim) != 0) {
    //   INTELLI_ERROR(string("Failed to open HDF5 file: ") + conf.dataset_path);
    //   exit(EXIT_FAILURE);
    // }
    // dis = uniform_int_distribution<>(0, points_num - 1);
  } else {
    INTELLI_ERROR(string("Invalid vector source: ") + conf.vector_source);
    exit(EXIT_FAILURE);
  }

  // Insert vectors concurrently
  for (int i = 0; i < conf.insert_thread_count; i++) {
    int n = dis(gen);
    vector<float> vec(data_load + n * dim, data_load + (n + 1) * dim);

    insert_futures.emplace_back(pool.submit([&db, vec]() {
      return db.insert_vector(vec);
    }));
  }

  // Query nearest vectors concurrently
  for (int i = 0; i < conf.query_thread_count; i++) {
    int n = dis(gen);
    int k = conf.k_nearest;
    vector<float> vec(data_load + n * dim, data_load + (n + 1) * dim);

    query_futures.emplace_back(pool.submit([&db, vec, k]() {
      return db.query_nearest_vectors(vec, k);
    }));
  }

  // Wait for inserts to complete
  for (auto &insert_future: insert_futures) {
    bool res = insert_future.get();
    if (!res) {
      INTELLI_ERROR("Insert operation failed.");
    }
  }

  // Wait for queries to complete
  for (auto &query_future: query_futures) {
    auto res = query_future.get();
    if (res.empty()) {
      INTELLI_ERROR("Query operation returned empty results.");
    }
  }

  pool.shutdown();
}
