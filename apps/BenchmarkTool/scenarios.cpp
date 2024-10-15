/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Junyao Dong
 *  Created on: 2024/10/15 21:23:58
 * Description: [Provide description here]
 */
#include "scenarios.hpp"
#include <Utils/logging.hpp>
#include <Utils/thread_pool.hpp>
#include <Utils/file_loader.hpp>
#include <toml.hpp>

#include <vector>
#include <fstream>
#include <random>

using namespace std;

const map<string, function<void(VectorDB &db, ScenarioConfig &conf)>> scenarios = { 
  {"insert", insert_scenario},
  {"query", query_scenario},
  {"multi_query_insert", multi_query_insert_scenario}
};

const map<std::string, std::string> supported_index = {
	{"hnsw", "xxx"}, 
	{"concurrent_hnsw", "xxx"}, 
	// others
};

ScenarioConfig::ScenarioConfig(const string& conf) {
    load(conf);
}

void ScenarioConfig::load(const string& conf) { 
	ifstream config_file(conf);

	if (!config_file.is_open()) { 
		INTELLI_ERROR(string("Failed to open config file: ") + conf);
		exit(EXIT_FAILURE); 
	}

	const auto data = toml::parse(config_file);

	scenario_name = toml::find_or<string>(data, "scenario_name", "insert_scenario");

	index_type = toml::find_or<string>(data, "index_type", "hnsw");
	if (supported_index.find(index_type) == supported_index.end()) { 
		INTELLI_ERROR("Error: Unsupported index type \"" + index_type + "\".");
		exit(EXIT_FAILURE); 
	}

	query_thread_count = toml::find_or<int>(data, "query_thread_count", 1);
	insert_thread_count = toml::find_or<int>(data, "insert_thread_count", 0);
	timeout_in_sec = toml::find_or<int>(data, "timeout_in_sec", 10);
  dataset_path = toml::find<string>(data, "dataset_path");
  vector_source = toml::find<string>(data, "vector_source");
  k_nearest = toml::find_or<size_t>(data, "k_nearest", 3);
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

void insert_vector_task(VectorDB &db, vector<float> &vec) {
  db.insert_vector(vec);
}

void query_vector_task(VectorDB &db, vector<float> &vec, size_t k) {
  db.query_nearest_vectors(vec, k);
}

void multi_query_insert_scenario(VectorDB &db, ScenarioConfig &conf) { 

  float* data_load = NULL;
  unsigned points_num, dim;
  
  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<> dis;

  std::vector<std::future<void>> insert_futures;
  std::vector<std::future<void>> query_futures;

  ThreadPool pool(conf.query_thread_count + conf.insert_thread_count);
  pool.init();

  if (conf.vector_source == "fvecs") {
    // load data from fvecs
    if (load_fvecs_data(conf.dataset_path, data_load, points_num, dim) != 0) { 
      INTELLI_ERROR(string("Failed to open fvevs file: ") + conf.dataset_path);
    }
    dis = std::uniform_int_distribution<>(0, points_num - 1);
  } else if (conf.vector_source == "hdf5") {

  } else { 
    INTELLI_ERROR(string("Invalid vector source: ") + conf.dataset_path);
    exit(EXIT_FAILURE);
  }
  
  for (int i = 0; i < conf.insert_thread_count; i++) { 
    int n = dis(gen);
    vector<float> vec(data_load + n * dim, data_load + (n + 1) * dim);

    insert_futures.emplace_back(pool.submit(insert_vector_task, ref(db), ref(vec)));
  } 

  for (int i = 0; i < conf.query_thread_count; i++) { 
    int n = dis(gen);
    vector<float> vec(data_load + n * dim, data_load + (n + 1) * dim);

    query_futures.emplace_back(pool.submit(query_vector_task, ref(db), ref(vec), conf.k_nearest));
  }

  // auto res = insert_futures[0].get();

}

