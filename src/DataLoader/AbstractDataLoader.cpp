/*
* Copyright (C) 2024 by the INTELLI team
 * Created on: 2024/10/9
 * Description: [Provide description here]
 */
#include <DataLoader/AbstractDataLoader.hpp>
using namespace std;

bool CANDY::AbstractDataLoader::setConfig(INTELLI::ConfigMapPtr cfg) {
  assert(cfg);

  return true;
}

torch::Tensor CANDY::AbstractDataLoader::getData() {
  return torch::rand({1, 1});
}

torch::Tensor CANDY::AbstractDataLoader::getQuery() {
  return torch::rand({1, 1});
}


int64_t CANDY::AbstractDataLoader::size() {
  auto ru = getData();
  return ru.size(0);
}