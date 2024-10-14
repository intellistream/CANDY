/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Junyao Dong
 * Created on: 2024/10/14
 * Description: [Provide description here]
 */

#include "benchmark_config.hpp"
#include <Utils/logging.hpp>

#include <fstream>
#include <iostream>
#include <map>
#include <cstdlib>

using namespace std;

const std::map<std::string, std::string> supported_index = {
	{"hnsw", "Hierarchical Navigable Small World"}, // naive hnsw from hnswlib
	{"concurrent_hnsw", "Inverted File"}, // concurrent hnsw 
	// others
};

BenchmarkConfig::BenchmarkConfig(const string& conf) {
    load(conf);
}

void BenchmarkConfig::load(const string& conf) { 
	ifstream config_file(conf);

	if (!config_file.is_open()) { 
		INTELLI_ERROR(string("Failed to open config file: ") + conf);
		exit(EXIT_FAILURE); 
	}

	const auto data = toml::parse(config_file);

	index_type = toml::find_or<string>(data, "index_type", "hnsw");
	if (supported_index.find(index_type) == supported_index.end()) { 
		INTELLI_ERROR("Error: Unsupported index type \"" + index_type + "\".");
		exit(EXIT_FAILURE); 
	}

	query_thread_count = toml::find_or<int>(data, "query_thread_count", 1);
	insert_thread_count = toml::find_or<int>(data, "insert_thread_count", 0);
	timeout_in_sec = toml::find_or<int>(data, "timeout_in_sec", 10);
}