#ifndef CANDY_VECTOR_DB_VECTOR_DATABASE_HPP
#define CANDY_VECTOR_DB_VECTOR_DATABASE_HPP

#include <candy_vector_db/storage_engine.hpp>
#include <candy_vector_db/indexing/anns.hpp>
#include <candy_vector_db/query_executor.hpp>
#include <memory>

namespace candy {

class VectorDatabase {
public:
  VectorDatabase(const std::string& storagePath);

  // Add a vector
  void addVector(const std::shared_ptr<VectorRecord>& record);

  // Remove a vector
  void removeVector(const std::string& id);

  // Execute a k-NN query
  std::vector<std::shared_ptr<VectorRecord>> executeKNN(const VectorData& query, size_t k);

private:
  std::shared_ptr<StorageEngine> storageEngine;
  std::shared_ptr<ANNS> anns;
  std::shared_ptr<QueryExecutor> queryExecutor;
};

} // namespace candy

#endif // CANDY_VECTOR_DB_VECTOR_DATABASE_HPP
