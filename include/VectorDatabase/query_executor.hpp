#pragma once

#include "storage_engine.hpp"
#include <VectorDatabase/indexing/anns.hpp>

class QueryExecutor {
public:
  QueryExecutor(StorageEngine *storage, ANNS *index) : storage(storage), index(index) {}

  // k-NN search using the index
  std::vector<int> knnSearch(const std::vector<float> &query, int k) {
    return index->knnSearch(query, k);
  }

private:
  StorageEngine *storage;
  ANNS *index;
};
