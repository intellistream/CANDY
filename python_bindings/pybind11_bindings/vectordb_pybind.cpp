/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/11
 * Description: [Provide description here]
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Core/vector_db.hpp>  // Corrected include to match header guard and existing header file
#include <Algorithms/KNN/KNNSearch.hpp>

namespace py = pybind11;

PYBIND11_MODULE(pycandy, m) {
  py::class_<ANNSBase, std::shared_ptr<ANNSBase>>(m, "ANNSBase")
      .def("insert_tensor", &ANNSBase::insertTensor)
      .def("search_tensor", &ANNSBase::searchTensor);

  py::class_<VectorDB>(m, "VectorDB")
      .def(py::init([](size_t dimensions, const std::string &search_algorithm) {
          std::shared_ptr<ANNSBase> algorithm;
          if (search_algorithm == "knnsearch") {
              algorithm = std::make_shared<KnnSearch>(dimensions);
          } else {
              throw std::invalid_argument("Unsupported search algorithm: " + search_algorithm);
          }
          return std::make_unique<VectorDB>(dimensions, algorithm);
      }), py::arg("dimensions"), py::arg("search_algorithm") = "knnsearch")

      .def("insert_tensor", &VectorDB::insert_tensor, py::arg("tensor"))

      .def("query_nearest_tensors", [](const VectorDB &self, const torch::Tensor &query_tensor, size_t k) {
          return self.query_nearest_tensors(query_tensor, k);
      }, py::arg("query_tensor"), py::arg("k"))

      .def("remove_tensor", &VectorDB::remove_tensor, py::arg("id"))

      .def("update_tensor", &VectorDB::update_tensor, py::arg("id"), py::arg("tensor"))

      .def("start_streaming", &VectorDB::start_streaming)

      .def("stop_streaming", &VectorDB::stop_streaming)

      .def("insert_streaming_tensor", &VectorDB::insert_streaming_tensor, py::arg("tensor"))

      .def("process_streaming_queue", &VectorDB::process_streaming_queue);
}