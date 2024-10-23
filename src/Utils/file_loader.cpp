/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Junyao Dong
 * Created on: 2024/10/15
 * Description: [Provide description here]
 */
#include <Utils/file_loader.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>

using namespace std;

int load_fvecs_data(string filename, float*& data, unsigned& num, unsigned& dim) {  
  ifstream in(filename, ios::binary);
  if (!in.is_open()) {
    return -1;
  }

  in.read((char*)&dim, 4);	// vector dimension
  in.seekg(0, ios::end);	// cursor to end of file
  ios::pos_type ss = in.tellg();	// get file size (how many bytes)
  size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim + 1) / 4);	// count of data
  data = new float[(size_t)num * (size_t)dim];

  in.seekg(0, ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, ios::cur);	
    in.read((char*)(data + i * dim), dim * 4);	
  }

  in.close();
  return 0;
}

int load_ivecs_data(string filename, int*& data, unsigned& num, unsigned& dim) {    
  ifstream in(filename, ios::binary);
  if (!in.is_open()) {
    return -1;
  }

  in.read((char*)&dim, 4);	// vector dimension
  in.seekg(0, ios::end);	// cursor to end of file
  ios::pos_type ss = in.tellg();	// get file size (how many bytes)
  size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim + 1) / 4);	// count of data
  data = new int[(size_t)num * (size_t)dim];

  in.seekg(0, ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, ios::cur);	
    in.read((char*)(data + i * dim), dim * 4);	
  }

  in.close();
  return 0;
}