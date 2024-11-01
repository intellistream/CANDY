/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Junyao Dong
 * Created on: 2024/10/16
 * Description: [Provide description here]
 */

#include <Utils/config_parser.hpp>
#include <Utils/logging.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

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

int ConfigParser::parse_ini(const string& fname) {
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

      try {
        if (value.find('.') != string::npos) {
          conf[key] = stof(value);
        } else {
          conf[key] = stoi(value);
        }
      } catch (...) {
        conf[key] = value;
      }
    }
  }
  return 0;
}

int ConfigParser::parse_csv(const string& fname) {
  ifstream file(fname);
  if (!file.is_open()) {
    throw runtime_error("Unable to open file: " + fname);
    return -1;
  }

  string line;
  while (getline(file, line)) {
    vector<string> cols;
    spilt(line, ",", cols);
    if (cols.size() >= 3) {
      istringstream iss(cols[1]);
      if (cols[2] == "U64" || cols[2] == "U64\r" || cols[2] == "I64" ||
          cols[2] == "I64\r" || cols[2] == "Int" || cols[2] == "Int\r") {
        int value;
        iss >> value;
        conf[cols[0]] = value;
      } else if (cols[2] == "Double" || cols[2] == "Double\r") {
        double value;
        iss >> value;
        conf[cols[0]] = static_cast<float>(value);
      } else if (cols[2] == "Float" || cols[2] == "Float\r") {
        float value;
        iss >> value;
        conf[cols[0]] = value;
      } else if (cols[2] == "String" || cols[2] == "String\r") {
        conf[cols[0]] = cols[1];
      }
    }
  }
  return 0;
}

string ConfigParser::get_string(const string& key,
                                const string& default_value) const {
  auto it = conf.find(key);
  if (it != conf.end()) {
    if (holds_alternative<string>(it->second)) {
      return get<string>(it->second);
    }
  }
  INTELLI_ERROR("Invalid string value for key: " + key);
  return default_value;
}

int ConfigParser::get_int(const string& key, int default_value) const {
  auto it = conf.find(key);
  if (it != conf.end()) {
    if (holds_alternative<int>(it->second)) {
      return get<int>(it->second);
    }
  }
  INTELLI_ERROR("Invalid int value for key: " + key);
  return default_value;
}

float ConfigParser::get_float(const string& key, float default_value) const {
  auto it = conf.find(key);
  if (it != conf.end()) {
    if (holds_alternative<float>(it->second)) {
      return get<float>(it->second);
    }
  }
  INTELLI_ERROR("Invalid float value for key: " + key);
  return default_value;
}

template void ConfigParser::edit<int>(const string& key, const int& value);

template void ConfigParser::edit<float>(const string& key, const float& value);

template void ConfigParser::edit<string>(const string& key,
                                         const string& value);