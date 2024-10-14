/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Junyao Dong
 * Created on: 2024/10/14
 * Description: [Provide description here]
 */
#include <toml.hpp>
#include <string>
#include <iostream>
#include <map>

using namespace std;

extern const map<string, string> supported_index;

class BenchmarkConfig { 
public:
	BenchmarkConfig(const string& conf);

	void load(const string& conf);

	int get_read_thread_count() const;
	int get_write_thread_count() const;

private:
	int query_thread_count;
	int insert_thread_count;
	int timeout_in_sec;
	string index_type;
};