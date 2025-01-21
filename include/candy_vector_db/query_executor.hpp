#ifndef CANDY_VECTOR_DB_QUERY_EXECUTOR_HPP
#define CANDY_VECTOR_DB_QUERY_EXECUTOR_HPP

#include <candy_core/common/data_types.hpp>
#include <candy_vector_db/indexing/anns.hpp>
#include <candy_vector_db/storage_engine.hpp>
#include <memory>
#include <string>
#include <vector>

namespace candy {

class QueryExecutor {
public:
  QueryExecutor(std::shared_ptr<StorageEngine> storage,
                std::shared_ptr<ANNS> anns);

  // Execute a k-NN search query
  std::vector<std::shared_ptr<VectorRecord>> executeKNN(const VectorData &query,
                                                        size_t k);

  // Add a vector
  void addVector(const std::shared_ptr<VectorRecord> &record);

  // Remove a vector
  void removeVector(const std::string &id);

private:
  std::shared_ptr<StorageEngine> storageEngine;
  std::shared_ptr<ANNS> anns;
};

} // namespace candy

#endif // CANDY_VECTOR_DB_QUERY_EXECUTOR_HPP
