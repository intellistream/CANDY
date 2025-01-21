#include <candy_vector_db/query_executor.hpp>

namespace candy {

QueryExecutor::QueryExecutor(std::shared_ptr<StorageEngine> storage, std::shared_ptr<ANNS> anns)
    : storageEngine(std::move(storage)), anns(std::move(anns)) {}

std::vector<std::shared_ptr<VectorRecord>> QueryExecutor::executeKNN(const VectorData& query, size_t k) {
  return anns->search(query, k);
}

void QueryExecutor::addVector(const std::shared_ptr<VectorRecord>& record) {
  storageEngine->add(record);
  anns->insert(record);
}

void QueryExecutor::removeVector(const std::string& id) {
  storageEngine->remove(id);
  anns->remove(id);
}

} // namespace candy