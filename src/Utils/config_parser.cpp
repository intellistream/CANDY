/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Junyao Dong
 * Created on: 2024/10/16
 * Description: [Provide description here]
 */

#include "../../include/Utils/config_parser.hpp"

#include <fstream>
#include <vector>
#include <assert.h>
#include <regex>

using namespace std;

string& strim(string &s, const string & del) {
  if (s.empty()) {
    return s;
  }

  s.erase(0,s.find_first_not_of(del));
  s.erase(s.find_last_not_of(del) + 1);
  return s;
}

vector<string> Split(const string & input, const string& regex) {
  // passing -1 as the submatch index parameter performs splitting
  std::regex re(regex);
  sregex_token_iterator first{input.begin(), input.end(), re, -1}, last;
  return {first, last};
}

ConfigParser::ConfigParser() {

}

ConfigParser::~ConfigParser() {
  auto it = m_config_data.begin();
  for (; it != m_config_data.end(); it++) {
    map<string, string> * sec = it->second;
    if (sec) {
      delete sec;
      it->second = NULL;
    }
  }
  m_config_data.clear();
}

bool ConfigParser::parser(const string & file_path)
{
  ifstream input(file_path);
  if (!input) {
      return false;
  }

  try {
    string cur_sec;
    for (string line; getline(input, line); ) {
      if (line.find_first_of("#") != string::npos) {
        line.erase(line.find_first_of("#"));
      }
      if (line.empty()) continue;

      if (line[0] == '[')  {
        if (line[line.size()-1] != ']') {
          throw runtime_error("section format error");
        }
        map<string, string> *sec = new map<string, string>();
        if (sec == NULL) {
          throw runtime_error("run out of memory");
        }
        string sectionName = string(line, 1, line.size()- 2);
        m_config_data.insert(make_pair(sectionName, sec));
        cur_sec = sectionName; 
      } else {
        vector<string> kv = Split(line, "=");
        if (kv.size() != 2) {
          throw runtime_error("ini format error");
        }
        string k = strim(kv[0], " ");
        string v = strim(kv[1], " ");
        if (cur_sec.empty()) {
          throw runtime_error("lack of section");
        }
        auto it = m_config_data.find(cur_sec);
        assert(it != m_config_data.end());
        it->second->insert(make_pair(k,v));
      }
    }
    input.close();
    return true;
  } catch(runtime_error e)  {
    auto it = m_config_data.begin();
    for (; it != m_config_data.end(); it++) {
      map<string, string> * sec = it->second;
      if (sec) {
        delete sec;
        it->second = NULL;
      }
    }
    m_config_data.clear();
    input.close();
    return false;
  }
}

bool ConfigParser::has_section(const string & sec) {
  return m_config_data.find(sec) != m_config_data.end() ? true : false;
}

int ConfigParser::get_sections(vector<string> & vec_sections) {
  for (auto it = m_config_data.begin(); it !=  m_config_data.end(); it++) {
    vec_sections.push_back(it->first);
  }
  return 0;
}

int ConfigParser::get_keys(const string & sec, vector<string> & keys) {
  auto it = m_config_data.find(sec);
  if (it != m_config_data.end()) {
    map<string, string> * psec = it->second;
    for (auto it2 = psec->begin(); it2 !=  psec->end(); it2++) {
      keys.push_back(it2->first);
    }
  }
  return 0;
}

const map<string, string> *ConfigParser::get_section_config(const string & sec) {
  auto it = m_config_data.find(sec);
  if (it != m_config_data.end()) {
    return it->second;
  }
  return NULL;
}

string ConfigParser::get_config(const string & sec, const string & key) {
  const map<string, string> * psec = get_section_config(sec);
  if (psec) {
    auto it = psec->find(key);
    if (it != psec->end()) {
      return it->second;
    }
  }
  return "";
}

string ConfigParser::get_def_config(const string & sec, const string & key, const string & def)
{
  const map<string, string> * psec = get_section_config(sec);
  if (psec) {
    auto it = psec->find(key);
    if (it != psec->end()) {
      return it->second;
    }
  }
  return def;
}