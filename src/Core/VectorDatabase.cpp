/*
* Copyright (C) 2024 by the INTELLI team
 * Created by: jjzhao
 * Created on: 2024/11/24
 * Description: [Provide description here]
 */

#include <Core/VectorDatabase.hpp>
VectorDatabase::VectorDatabase() {

}
VectorDatabase::~VectorDatabase() {

}
bool VectorDatabase::insert_tensor_rawid (const torch::Tensor&tensor, int rawId) {
  auto insert_container = torch::zeros({1, tensor.size(0)});
  insert_container[0] = tensor;
  return anns_algorithm.insertTensorWithRawId(insert_container, rawId);
}
std::vector<int> VectorDatabase::delete_tensor(const torch::Tensor& tensor, size_t k) {
  return anns_algorithm.deleteTensor(tensor, k);
}
std::vector<int> VectorDatabase::search_tensor(const torch::Tensor& query_tensor, size_t k) {
  return anns_algorithm.searchTensor(query_tensor, k);
}
std::string VectorDatabase::displayStore() {
  return anns_algorithm.storage_engine.display();
}
