#ifndef CANDY_VECTOR_DB_ANNS_HPP
#define CANDY_VECTOR_DB_ANNS_HPP

#include <candy_core/common/data_types.hpp>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace candy {

class ANNS {
public:
  // Insert a vector into the index
  void insert(const std::shared_ptr<VectorRecord> &record);

  // Search for the k-nearest neighbors
  std::vector<std::shared_ptr<VectorRecord>> search(const VectorData &query,
                                                    size_t k);

  // Delete a vector from the index
  void remove(const std::string &id);

private:
  // Internal in-memory index (ID -> VectorRecord)
  std::unordered_map<std::string, std::shared_ptr<VectorRecord>> index;
};

} // namespace candy

#endif // CANDY_VECTOR_DB_ANNS_HPP
