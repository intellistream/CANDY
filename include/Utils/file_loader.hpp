/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Junyao Dong
 * Created on: 2024/10/15
 * Description: [Provide description here]
 */

#ifndef INTELLISTREAM_SRC_UTILS_FILE_LOADER_HPP_
#define INTELLISTREAM_SRC_UTILS_FILE_LOADER_HPP_

#include <string>
#include <vector>

using namespace std;

int load_fvecs_data(std::string filename, float*& data, unsigned& num, unsigned& dim);

int load_ivecs_data(const char* filename, vector<vector<unsigned>>& results, 
  unsigned &num, unsigned &dim);

#endif //INTELLISTREAM_SRC_UTILS_FILE_LOADER_HPP_