/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Junyao Dong
 * Created on: 2024/10/16
 * Description: [Provide description here]
 */

#include <Utils/config_parser.hpp>

#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

void trim(string& s) {
  size_t start = s.find_first_not_of(" \t\n\r");
  if (start != string::npos) {
    s = s.substr(start);
  }
  size_t end = s.find_last_not_of(" \t\n\r");
  if (end != string::npos) {
    s = s.substr(0, end + 1);
  } else {
    s.clear();
  }
}

int ConfigParser::parse(const string& fname) { 
  ifstream file(fname);
  if (!file.is_open()) {
    throw runtime_error("Unable to open file: " + fname);
    return -1;
  }

  string line;
  while (getline(file, line)) {
    trim(line);

    if (line.empty() || line[0] == ';' || line[0] == '#') {
      continue;
    }

    size_t equal_pos = line.find('=');
    if (equal_pos != string::npos) {
      string key = line.substr(0, equal_pos);
      string value = line.substr(equal_pos + 1);
      trim(key);
      trim(value);

      conf[key] = value;
    }
  }
  return 0;
}

string ConfigParser::get_string(const string& key, const string& default_value) const {
  auto it = conf.find(key);
  return it != conf.end() ? it->second : default_value;
}

int ConfigParser::get_int(const string& key, int default_value) const {
  auto it = conf.find(key);
  if (it != conf.end()) {
    try {
      return stoi(it->second);
    } catch (...) {
      throw invalid_argument("Invalid int value for key: " + key);
    }
  }
  return default_value;
}

float ConfigParser::get_float(const string& key, float default_value) const {
  auto it = conf.find(key);
  if (it != conf.end()) {
    try {
      return stof(it->second);
    } catch (...) {
      throw invalid_argument("Invalid float value for key: " + key);
    }
  }
  return default_value;
}