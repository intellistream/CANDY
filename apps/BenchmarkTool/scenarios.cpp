/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Junyao Dong
 * Created on: 2024/10/16 14:12:04
 * Description: [Provide description here]
 */
#include "scenarios.hpp"
#include <Utils/logging.hpp>
#include <Utils/thread_pool.hpp>
#include <Utils/file_loader.hpp>
#include <Utils/config_parser.hpp>

#include <vector>
#include <fstream>
#include <random>
#include <iterator>

using namespace std;

const map<string, function<void(VectorDB &db, ScenarioConfig &conf)>> scenarios = { 
  {"insert", insert_scenario},
  {"query", query_scenario},
  {"multi_query_insert", multi_query_insert_scenario}
};

// TODO
const map<std::string, std::string> supported_index = {
	{"hnsw", "xxx"}, 
	{"concurrent_hnsw", "xxx"}, 
	// others
};

ScenarioConfig::ScenarioConfig(const string& conf_path) {
    load(conf_path);
}

void ScenarioConfig::load(const string& conf_path) { 
  ConfigParser parser;
  parser.parse(conf_path);

  scenario_name = parser.get_string("scenario_name");
  index_type = parser.get_string("index_type");
  vector_source = parser.get_string("vector_source");
  dataset_path = parser.get_string("dataset_path");
  k_nearest =  parser.get_int("k_nearest");
  query_thread_count = parser.get_int("query_thread_count", 1);
  insert_thread_count = parser.get_int("insert_thread_count", 0);
  timeout_in_sec = parser.get_int("timeout_in_sec", 10);
  dimension = parser.get_int("dimension");
}

void insert_scenario(VectorDB &db, ScenarioConfig &conf) { 
  for (int i = 0; i < 10000; ++i) {
    vector<float> new_vector = {static_cast<float>(i), static_cast<float>(i + 1), static_cast<float>(i + 2)};
    db.insert_vector(new_vector);
  }
}

void query_scenario(VectorDB &db, ScenarioConfig &conf) { 
  for (int i = 0; i < 10000; ++i) {
    vector<float> new_vector = {static_cast<float>(i), static_cast<float>(i + 1), static_cast<float>(i + 2)};
    db.insert_vector(new_vector);
  }
}

void multi_query_insert_scenario(VectorDB &db, ScenarioConfig &conf) { 
  float* data_load = NULL;
  unsigned points_num, dim;
  
  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<> dis;

  std::vector<std::future<bool>> insert_futures;
  std::vector<std::future<std::vector<std::vector<float>>>> query_futures;

  ThreadPool pool(conf.query_thread_count + conf.insert_thread_count);
  pool.init();

  if (conf.vector_source == "fvecs") {
    // load data from fvecs
    if (load_fvecs_data(conf.dataset_path, data_load, points_num, dim) != 0) { 
      INTELLI_ERROR(string("Failed to open fvevs file: ") + conf.dataset_path);
      exit(EXIT_FAILURE);
    }
    dis = std::uniform_int_distribution<>(0, points_num - 1);
  } else if (conf.vector_source == "hdf5") {
    // TODO
  } else { 
    INTELLI_ERROR(string("Invalid vector source: ") + conf.dataset_path);
    exit(EXIT_FAILURE);
  }
  
  for (int i = 0; i < conf.insert_thread_count; i++) { 
    int n = dis(gen);
    vector<float> vec(data_load + n * dim, data_load + (n + 1) * dim);

    insert_futures.emplace_back(pool.submit([&db, &vec]() {
      return db.insert_vector(vec);
    }));
  } 

  for (int i = 0; i < conf.query_thread_count; i++) { 
    int n = dis(gen);
    int k = conf.k_nearest;
    vector<float> vec(data_load + n * dim, data_load + (n + 1) * dim);

    query_futures.emplace_back(pool.submit([&db, &vec, k]() {
      return db.query_nearest_vectors(vec, k);
    }));
  }

  // TODO 
  // Simple return just for test
  for (int i = 0; i < conf.insert_thread_count; i++) {
    auto res = insert_futures[i].get();
    std::cout << "test insert result: " << res << std::endl;
  }

  for (int i = 0; i < conf.query_thread_count; i++) {
    auto res = query_futures[i].get();
    std::cout << "test query result: ";
    for (auto r : res) 
      for (auto rr : r)
        std::cout << rr << " ";
    
    std::cout << std::endl;
  }

  pool.shutdown();
}

