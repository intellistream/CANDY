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
#include <variant>
#include <memory>
#include <vector>

using namespace std;

class ConfigParser {
protected:
  static void spilt(const string s, const string &c, vector<string> &v) {
    string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while (string::npos != pos2) {
      v.push_back(s.substr(pos1, pos2 - pos1));
      pos1 = pos2 + c.size();
      pos2 = s.find(c, pos1);
    }
    if (pos1 != s.length()) {
      v.push_back(s.substr(pos1));
    }
  }
  unordered_map<string, variant<int, float, string>> conf;
public:
  ConfigParser() = default;

  ~ConfigParser() = default;

  ConfigParser(const ConfigParser &other) {
    load_config(other);
  }

  void load_config(const ConfigParser &other) {
    conf = other.conf;
  }

  template <typename T>
  void edit(const string &key, const T &value) {
    conf[key] = value;
  }

  int parse_ini(const string& fname);

  int parse_csv(const string& fname);

  string get_string(const string& key, const string& default_value = "") const;

  int get_int(const string& key, int default_value = 0) const;

  float get_float(const string& key, float default_value = 0.0f) const;
};

typedef std::shared_ptr<ConfigParser> ConfigParserPtr;
#define  newConfigParser make_shared<ConfigParser>

#endif //INTELLISTREAM_SRC_UTILS_CONFIG_PARSER_HPP_