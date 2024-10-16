/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Junyao Dong
 * Created on: 2024/10/16
 * Description: [Provide description here]
 */

#ifndef INTELLISTREAM_SRC_UTILS_CONFIG_PARSER_HPP_
#define INTELLISTREAM_SRC_UTILS_CONFIG_PARSER_HPP_

#include <string>
#include <vector>
#include <map>

using namespace std;

class ConfigParser {
public:
  ConfigParser();
  ~ConfigParser();

public:
  bool parser(const string & file_path);
  bool has_section(const string & section);
  int get_sections(vector<string> & vec_sections);
  int get_keys(const string & str_section, vector<string> & vec);
  const map<string, string> * get_section_config(const string & section);
  string get_config(const string & section, const string & key);
  string get_def_config(const string & section, const string & key, const string & def);
private:
  map<string, map<string, string> *> m_config_data;
};

#endif //INTELLISTREAM_SRC_UTILS_CONFIG_PARSER_HPP_