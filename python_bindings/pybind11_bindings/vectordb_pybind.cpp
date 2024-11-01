/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/11
 * Description: [Provide description here]
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <API/vectordb_stream.hpp>
#include <Algorithms/knn_search.hpp>
#include <Core/vector_db.hpp>  // Corrected include to match header guard and existing header file

namespace py = pybind11;

PYBIND11_MODULE(pycandy, m) {
  py::class_<SearchAlgorithm, std::shared_ptr<SearchAlgorithm>>(
      m, "SearchAlgorithm");

  py::class_<VectorDB>(m, "VectorDB")
      .def(py::init([](size_t dimensions, const std::string& search_algorithm) {
             std::shared_ptr<SearchAlgorithm> algorithm;
             if (search_algorithm == "knnsearch") {
               algorithm = std::make_shared<KnnSearch>(dimensions);
             } else {
               throw std::invalid_argument("Unsupported search algorithm: " +
                                           search_algorithm);
             }
             return std::make_unique<VectorDB>(dimensions, algorithm);
           }),
           py::arg("dimensions"), py::arg("search_algorithm") = "knnsearch")
      .def("insert_vector", &VectorDB::insert_vector)
      .def(
          "query_nearest_vectors",
          [](const VectorDB& self, const std::vector<float>& query_vec,
             size_t k) { return self.query_nearest_vectors(query_vec, k); },
          py::arg("query_vec"), py::arg("k"))
      .def("remove_vector", &VectorDB::remove_vector, py::arg("id"))
      .def("update_vector", &VectorDB::update_vector, py::arg("id"),
           py::arg("vector"))
      .def("start_streaming", &VectorDB::start_streaming)
      .def("stop_streaming", &VectorDB::stop_streaming)
      .def("insert_streaming_vector", &VectorDB::insert_streaming_vector)
      .def("process_streaming_queue", &VectorDB::process_streaming_queue);

  py::class_<VectorDBStream>(m, "VectorDBStream")
      .def(py::init<>())
      .def_static("from", &VectorDBStream::from, py::arg("input_vectors"))
      .def("map", &VectorDBStream::map, py::arg("func"))
      .def("filter", &VectorDBStream::filter, py::arg("predicate"))
      .def("map_to_embedding", &VectorDBStream::map_to_embedding,
           py::arg("embedding_func"))
      .def("to_sink", &VectorDBStream::to_sink, py::arg("vector_db"))
      .def("query_nearest", &VectorDBStream::query_nearest,
           py::arg("query_vec"), py::arg("k"));
}