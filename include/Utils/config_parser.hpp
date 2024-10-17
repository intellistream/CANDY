/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Junyao Dong
 * Created on: 2024/10/16
 * Description: [Provide description here]
 */

#ifndef INTELLISTREAM_SRC_UTILS_CONFIG_PARSER_HPP_
#define INTELLISTREAM_SRC_UTILS_CONFIG_PARSER_HPP_

#include <string>
#include <unordered_map>

using namespace std;

class ConfigParser { 
public:
  unordered_map<string, string> conf;

  int parse(const string& fname);

  string get_string(const string& key, const string& default_value = "") const;

  int get_int(const string& key, int default_value = 0) const;

  float get_float(const string& key, float default_value = 0.0f) const;
};

#endif //INTELLISTREAM_SRC_UTILS_CONFIG_PARSER_HPP_