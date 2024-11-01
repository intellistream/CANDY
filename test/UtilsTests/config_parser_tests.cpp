/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Junyao Dong
 * Created on: 2024/10/17
 * Description: [Provide description here]
 */
#include <Utils/config_parser.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdio>
#include <fstream>

using namespace std;

void write_ini(const string& fname) {
  ofstream file(fname);
  if (!file.is_open()) {
    return;
  }

  file << "; comment1\n";
  file << "# comment2\n";
  file << "key1 = value1\n";
  file << "key2 = 5\n";
  file << "key3 = 1.8\n";
}

TEST_CASE("ConfigParser: Parsing and Retrieving Values From INI File") {
  string fname = "test_config.ini";
  write_ini(fname);

  ConfigParser parser;
  parser.parse_ini("test_config.ini");

  SECTION("Get string value from config") {
    REQUIRE(parser.get_string("key1", "default_value") == "value1");
  }

  SECTION("Get int value from config") {
    REQUIRE(parser.get_int("key2", 0) == 5);
  }

  SECTION("Get float value from config") {
    REQUIRE(fabs(parser.get_float("key3", 0.0f) - 1.8f) < 1e-5);
  }

  SECTION("Get default values for non-existing keys") {
    REQUIRE(parser.get_string("non_existing_key", "default_value") ==
            "default_value");
    REQUIRE(parser.get_int("non_existing_key", 10) == 10);
    REQUIRE(fabs(parser.get_float("non_existing_key", 1.23f) - 1.23f) < 1e-5);
  }

  remove(fname.c_str());
}

TEST_CASE("ConfigParser: Parsing and Retrieving Values From CSV File") {
  string fname = "test_config.csv";
  ofstream file(fname);
  if (file.is_open()) {
    file << "key1,value1,String\n";
    file << "key2,5,Int\n";
    file << "key3,1.8,Float\n";
    file.close();
  }

  ConfigParser parser;
  parser.parse_csv(fname);

  SECTION("Get string value from config") {
    REQUIRE(parser.get_string("key1", "default_value") == "value1");
  }

  SECTION("Get int value from config") {
    REQUIRE(parser.get_int("key2", 0) == 5);
  }

  SECTION("Get float value from config") {
    REQUIRE(fabs(parser.get_float("key3", 0.0f) - 1.8f) < 1e-5);
  }

  SECTION("Get default values for non-existing keys") {
    REQUIRE(parser.get_string("non_existing_key", "default_value") ==
            "default_value");
    REQUIRE(parser.get_int("non_existing_key", 10) == 10);
    REQUIRE(fabs(parser.get_float("non_existing_key", 1.23f) - 1.23f) < 1e-5);
  }

  remove(fname.c_str());
}