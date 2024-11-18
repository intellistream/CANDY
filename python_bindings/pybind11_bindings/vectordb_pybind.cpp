/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/11
 * Description: [Provide description here]
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <Algorithms/KNN/KNNSearch.hpp>
#include <Core/vector_db.hpp>  // Corrected include to match header guard and existing header file
namespace py = pybind11;

PYBIND11_MODULE(pycandy, m) {

  py::class_<CANDY_ALGO::ANNSBase, std::shared_ptr<CANDY_ALGO::ANNSBase>>(
      m, "ANNSBase")
      .def("insert_tensor", &CANDY_ALGO::ANNSBase::insertTensor)
      .def("search_tensor", &CANDY_ALGO::ANNSBase::searchTensor);

  py::class_<VectorDB>(m, "VectorDB")
      .def(py::init([](size_t dimensions, const std::string& search_algorithm) {
             std::shared_ptr<CANDY_ALGO::ANNSBase> algorithm;
             if (search_algorithm == "knnsearch") {
               algorithm = std::make_shared<CANDY_ALGO::KnnSearch>(dimensions);
               algorithm->setConfig(nullptr);
             } else {
               throw std::invalid_argument("Unsupported search algorithm: " +
                                           search_algorithm);
             }
             return std::make_unique<VectorDB>(dimensions, algorithm);
           }),
           py::arg("dimensions"), py::arg("search_algorithm") = "knnsearch")

      .def("insert_tensor", &VectorDB::insert_tensor, py::arg("tensor"))

      .def(
          "query_nearest_tensors",
          [](const VectorDB& self, const torch::Tensor& query_tensor,
             size_t k) { return self.query_nearest_tensors(query_tensor, k); },
          py::arg("query_tensor"), py::arg("k"))

      .def("remove_tensor", &VectorDB::remove_tensor, py::arg("tensor"))

      .def("update_tensor", &VectorDB::update_tensor, py::arg("old_tensor"),
           py::arg("new_tensor"))

      .def("start_streaming", &VectorDB::start_streaming)

      .def("stop_streaming", &VectorDB::stop_streaming)

      .def("insert_streaming_tensor", &VectorDB::insert_streaming_tensor,
           py::arg("tensor"))

      .def("process_streaming_queue", &VectorDB::process_streaming_queue);
}