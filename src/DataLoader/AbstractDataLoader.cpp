/*
* Copyright (C) 2024 by the INTELLI team
 * Created on: 2024/10/9
 * Description: [Provide description here]
 */
#include <DataLoader/AbstractDataLoader.hpp>
using namespace std;

bool CANDY_ALGO::AbstractDataLoader::setConfig(INTELLI::ConfigMapPtr cfg) {
  assert(cfg);

  return true;
}

torch::Tensor CANDY_ALGO::AbstractDataLoader::getData() {
  return torch::rand({1, 1});
}

torch::Tensor CANDY_ALGO::AbstractDataLoader::getQuery() {
  return torch::rand({1, 1});
}

torch::Tensor CANDY_ALGO::AbstractDataLoader::getDataAt(int64_t startPos, int64_t endPos) {
  auto ru = getData();
  return ru.slice(0, startPos, endPos).nan_to_num(0);
}

int64_t CANDY_ALGO::AbstractDataLoader::size() {
  auto ru = getData();
  return ru.size(0);
}