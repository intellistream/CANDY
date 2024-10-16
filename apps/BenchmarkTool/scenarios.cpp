/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Junyao Dong
 *  Created on: 2024/10/16 13:23:29
 * Description: [Provide description here]
 */
#include "scenarios.hpp"
#include <Utils/logging.hpp>
#include <Utils/thread_pool.hpp>
#include <Utils/file_loader.hpp>
#include <Utils/config_parser.hpp>

#include <vector>
#include <fstream>
#include <random>

using namespace std;

const map<string, function<void(VectorDB &db, ScenarioConfig &conf)>> scenarios = { 
  {"insert", insert_scenario},
  {"query", query_scenario},
  {"multi_query_insert", multi_query_insert_scenario}
};

const map<std::string, std::string> supported_index = {
	{"hnsw", "xxx"}, 
	{"concurrent_hnsw", "xxx"}, 
	// others
};

ScenarioConfig::ScenarioConfig(const string& conf_path) {
    load(conf_path);
}

void ScenarioConfig::load(const string& conf_path) { 
  ConfigParser *cp = new ConfigParser();
  bool res = cp->parser(conf_path);
  if (res) { 
    // base section 
    scenario_name = cp->get_config("base", "scenario_name");
    index_type = cp->get_config("base", "index_type");
    vector_source = cp->get_config("base", "vector_source");
    k_nearest = stoi(cp->get_config("base", "k_nearest"));
    query_thread_count = stoi(cp->get_def_config("base", "query_thread_count", "1"));
    insert_thread_count = stoi(cp->get_def_config("base", "insert_thread_count", "0"));
    timeout_in_sec = stoi(cp->get_def_config("base", "timeout_in_sec", "10"));

    // advanced section
  }
}

void insert_scenario(VectorDB &db, ScenarioConfig &conf) { 
  for (int i = 0; i < 10000; ++i) {
    vector<float> new_vector = {static_cast<float>(i), static_cast<float>(i + 1), static_cast<float>(i + 2)};
    db.insert_vector(new_vector);
  }
}

void query_scenario(VectorDB &db, ScenarioConfig &conf) { 
  for (int i = 0; i < 10000; ++i) {
    vector<float> new_vector = {static_cast<float>(i), static_cast<float>(i + 1), static_cast<float>(i + 2)};
    db.insert_vector(new_vector);
  }
}

void multi_query_insert_scenario(VectorDB &db, ScenarioConfig &conf) { 
  std::cout << "yyssy\n ";
  float* data_load = NULL;
  unsigned points_num, dim;
  
  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<> dis;

  std::vector<std::future<bool>> insert_futures;
  std::vector<std::future<std::vector<std::vector<float>>>> query_futures;

  ThreadPool pool(conf.query_thread_count + conf.insert_thread_count);
  pool.init();

  std::cout << "yyy\n ";
  if (conf.vector_source == "fvecs") {
    // load data from fvecs
    if (load_fvecs_data(conf.dataset_path, data_load, points_num, dim) != 0) { 
      INTELLI_ERROR(string("Failed to open fvevs file: ") + conf.dataset_path);
      exit(EXIT_FAILURE);
    }
    dis = std::uniform_int_distribution<>(0, points_num - 1);
  } else if (conf.vector_source == "hdf5") {

  } else { 
    INTELLI_ERROR(string("Invalid vector source: ") + conf.dataset_path);
    exit(EXIT_FAILURE);
  }
  
  std::cout << "yyy\n ";
  for (int i = 0; i < conf.insert_thread_count; i++) { 
    int n = dis(gen);
    vector<float> vec(data_load + n * dim, data_load + (n + 1) * dim);

    insert_futures.emplace_back(pool.submit([&db, &vec]() {
      return db.insert_vector(vec);
    }));
  } 

  // for (int i = 0; i < conf.query_thread_count; i++) { 
  //   int n = dis(gen);
  //   int k = conf.k_nearest;
  //   vector<float> vec(data_load + n * dim, data_load + (n + 1) * dim);

  //   query_futures.emplace_back(pool.submit([&db, &vec, k]() {
  //     return db.query_nearest_vectors(vec, k);
  //   }));
  // }

  auto res = insert_futures[0].get();
  std::cout << "xxx " << res;
}

