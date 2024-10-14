/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Junyao Dong
 * Created on: 2024/10/14
 * Description: [Provide description here]
 */
#include "scenarios.hpp"

#include <vector>

using namespace std;

map<string, function<void(VectorDB &db)>> scenarios = { 
  {"insert", insertScenario},
  {"query", queryScenario},
  {"multi_query_insert", multiQueryInsertScenario}
};

void insertScenario(VectorDB &db) { 
  for (int i = 0; i < 10000; ++i) {
    vector<float> new_vector = {static_cast<float>(i), static_cast<float>(i + 1), static_cast<float>(i + 2)};
    db.insert_vector(new_vector);
  }
}

void queryScenario(VectorDB &db) { 
  for (int i = 0; i < 10000; ++i) {
    vector<float> new_vector = {static_cast<float>(i), static_cast<float>(i + 1), static_cast<float>(i + 2)};
    db.insert_vector(new_vector);
  }
}

void multiQueryInsertScenario(VectorDB &db) { 
  
}

