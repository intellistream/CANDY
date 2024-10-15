/*
 *  Copyright (C):  2024 by the INTELLI team
 *  Created by: Junyao Dong
 *  Created on: 2024/10/15 15:52:52
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

	int get_read_thread_count() const;
	int get_write_thread_count() const;

	int query_thread_count;
	int insert_thread_count;
	int timeout_in_sec;
	string index_type;
	string scenario_name;
};

void insertScenario(VectorDB &db, ScenarioConfig &conf);

void queryScenario(VectorDB &db, ScenarioConfig &conf);

void multiQueryInsertScenario(VectorDB &db, ScenarioConfig &conf);

extern const map<string, function<void(VectorDB &db, ScenarioConfig &conf)>> scenarios;

extern const map<std::string, std::string> supported_index;