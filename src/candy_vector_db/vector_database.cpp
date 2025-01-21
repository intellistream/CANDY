// vector_database.cpp
#include <candy_vector_db/vector_database.hpp>

namespace candy {

VectorDatabase::VectorDatabase(const std::string &storagePath)
    : storageEngine(std::make_shared<StorageEngine>(storagePath)),
      anns(std::make_shared<ANNS>()),
      queryExecutor(std::make_shared<QueryExecutor>(storageEngine, anns)) {}

void VectorDatabase::addVector(const std::shared_ptr<VectorRecord> &record) {
  queryExecutor->addVector(record);
}

void VectorDatabase::removeVector(const std::string &id) {
  queryExecutor->removeVector(id);
}

std::vector<std::shared_ptr<VectorRecord>>
VectorDatabase::executeKNN(const VectorData &query, size_t k) {
  return queryExecutor->executeKNN(query, k);
}

} // namespace candy