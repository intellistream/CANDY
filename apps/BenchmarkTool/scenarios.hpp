/*
 *  Copyright (C):  2024 by the INTELLI team
 *  Created by: Junyao Dong
 *  Created on: 2024/10/15 18:41:51
 *  Description:
 */

#pragma once

#include <Core/vector_db.hpp>

#include <map>

using namespace std;

class ScenarioConfig { 
public:
	ScenarioConfig(const string& conf);

	void load(const string& conf);

	int query_thread_count;
	int insert_thread_count;
	int timeout_in_sec;
	string index_type;
	string scenario_name;
};

void insert_scenario(VectorDB &db, ScenarioConfig &conf);

void query_scenario(VectorDB &db, ScenarioConfig &conf);

void multi_query_insert_scenario(VectorDB &db, ScenarioConfig &conf);

extern const map<string, function<void(VectorDB &db, ScenarioConfig &conf)>> scenarios;

extern const map<std::string, std::string> supported_index;