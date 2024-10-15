/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Junyao Dong
 * Created on: 2024/10/14
 * Description: [Provide description here]
 */
#include "scenarios.hpp"
#include <Utils/logging.hpp>
#include <toml.hpp>

#include <vector>
#include <fstream>

using namespace std;

const map<string, function<void(VectorDB &db, ScenarioConfig &conf)>> scenarios = { 
  {"insert", insertScenario},
  {"query", queryScenario},
  {"multi_query_insert", multiQueryInsertScenario}
};

const map<std::string, std::string> supported_index = {
	{"hnsw", "xxx"}, // naive hnsw from hnswlib
	{"concurrent_hnsw", "xxx"}, // concurrent hnsw 
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

	scenario_name = toml::find_or<string>(data, "scenario_name", "insertScenario");

	index_type = toml::find_or<string>(data, "index_type", "hnsw");
	if (supported_index.find(index_type) == supported_index.end()) { 
		INTELLI_ERROR("Error: Unsupported index type \"" + index_type + "\".");
		exit(EXIT_FAILURE); 
	}

	query_thread_count = toml::find_or<int>(data, "query_thread_count", 1);
	insert_thread_count = toml::find_or<int>(data, "insert_thread_count", 0);
	timeout_in_sec = toml::find_or<int>(data, "timeout_in_sec", 10);
}

void insertScenario(VectorDB &db, ScenarioConfig &conf) { 
  for (int i = 0; i < 10000; ++i) {
    vector<float> new_vector = {static_cast<float>(i), static_cast<float>(i + 1), static_cast<float>(i + 2)};
    db.insert_vector(new_vector);
  }
}

void queryScenario(VectorDB &db, ScenarioConfig &conf) { 
  for (int i = 0; i < 10000; ++i) {
    vector<float> new_vector = {static_cast<float>(i), static_cast<float>(i + 1), static_cast<float>(i + 2)};
    db.insert_vector(new_vector);
  }
}

void multiQueryInsertScenario(VectorDB &db, ScenarioConfig &conf) { 
  
}

