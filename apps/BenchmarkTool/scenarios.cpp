/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Junyao Dong
 * Created on: 2024/10/14
 * Description: [Provide description here]
 */
#include "scenarios.hpp"
#include <Utils/logging.hpp>
#include <Utils/thread_pool.hpp>
#include <toml.hpp>

#include <vector>
#include <fstream>

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

void insert_vector_task(VectorDB &db, ScenarioConfig &conf) {
  vector<float> new_vector = {static_cast<float>(1000), static_cast<float>(1002), static_cast<float>(1003)};
  db.insert_vector(new_vector);
}

void query_vector_task(VectorDB &db, ScenarioConfig &conf) {
  vector<float> new_vector = {static_cast<float>(1000), static_cast<float>(1001), static_cast<float>(1002)};
  db.query_nearest_vectors(new_vector, 3);
}

void multi_query_insert_scenario(VectorDB &db, ScenarioConfig &conf) { 
  std::vector<std::future<void>> insert_futures;
  std::vector<std::future<void>> query_futures;

  ThreadPool pool(conf.query_thread_count + conf.insert_thread_count);
  pool.init();

  
  for (int i = 0; i < conf.insert_thread_count; i++) { 
    insert_futures.push_back(pool.submit(insert_vector_task, db, conf));
  } 

  for (int i = 0; i < conf.query_thread_count; i++) { 
    query_futures.push_back(pool.submit(query_vector_task, db, conf));
  }

  // auto res = insert_futures[0].get();

}

