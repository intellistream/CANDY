/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/11
 * Description: [Provide description here]
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <Algorithms/HNSW/hnsw.hpp>
#include <Algorithms/KNN/KNNSearch.hpp>
#include <Core/vector_db.hpp>  // Corrected include to match header guard and existing header file
#include <DataLoader/DataLoaderTable.hpp>
#include <Utils/ConfigMap.hpp>
namespace py = pybind11;
using namespace INTELLI;
using namespace std;

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

      .def(py::init([](size_t dimensions, const std::string& search_algorithm,
                       INTELLI::ConfigMapPtr cfg) {
             std::shared_ptr<CANDY_ALGO::ANNSBase> algorithm;
             if (search_algorithm == "knnsearch") {
               algorithm = std::make_shared<CANDY_ALGO::KnnSearch>(dimensions);
               algorithm->setConfig(cfg);  // 使用传入的配置
             } else if (search_algorithm == "hnswsearch") {
               algorithm = std::make_shared<CANDY_ALGO::HNSW>();
               algorithm->setConfig(cfg);  // 使用传入的配置
             } else {
               throw std::invalid_argument("Unsupported search algorithm: " +
                                           search_algorithm);
             }
             return std::make_unique<VectorDB>(dimensions, algorithm);
           }),
           py::arg("dimensions"), py::arg("search_algorithm") = "knnsearch",
           py::arg("config") = nullptr)

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

  // m.def("createDataLoader", &creatDataLoader,
  //     "A function to create new data loader by name tag");
  //
  // py::class_<CANDY_ALGO::AbstractDataLoader,
  //          std::shared_ptr<CANDY_ALGO::AbstractDataLoader>>(
  //   m, "AbstractDataLoader")
  //     .def(py::init<>())
  //     .def("setConfig", &CANDY_ALGO::AbstractDataLoader::setConfig)
  //     .def("getData", &CANDY_ALGO::AbstractDataLoader::getData)
  //     .def("getDataAt", &CANDY_ALGO::AbstractDataLoader::getDataAt)
  //     .def("getQuery", &CANDY_ALGO::AbstractDataLoader::getQuery);

  py::class_<CANDY_ALGO::AbstractDataLoader,
             std::shared_ptr<CANDY_ALGO::AbstractDataLoader>>(
      m, "AbstractDataLoader")
      .def(py::init<>())
      .def("setConfig", &CANDY_ALGO::AbstractDataLoader::setConfig)
      .def("getData", &CANDY_ALGO::AbstractDataLoader::getData)
      .def("getDataAt", &CANDY_ALGO::AbstractDataLoader::getDataAt)
      .def("getQuery", &CANDY_ALGO::AbstractDataLoader::getQuery);

  py::class_<CANDY_ALGO::DataLoaderTable,
             std::shared_ptr<CANDY_ALGO::DataLoaderTable>>(m, "DataLoaderTable")
      .def(py::init<>())
      .def("register_new_data_loader",
           &CANDY_ALGO::DataLoaderTable::registerNewDataLoader)
      .def("find_data_loader", &CANDY_ALGO::DataLoaderTable::findDataLoader);

  py::class_<ConfigMap, std::shared_ptr<ConfigMap>>(m, "ConfigMap")
      .def(py::init<>())  // Constructor
      .def("edit_u64",
           static_cast<void (ConfigMap::*)(const std::string&, uint64_t)>(
               &ConfigMap::edit),
           py::arg("key"), py::arg("value"))
      .def("edit_i64",
           static_cast<void (ConfigMap::*)(const std::string&, int64_t)>(
               &ConfigMap::edit),
           py::arg("key"), py::arg("value"))
      .def("edit_double",
           static_cast<void (ConfigMap::*)(const std::string&, double)>(
               &ConfigMap::edit),
           py::arg("key"), py::arg("value"))
      .def("edit_str",
           static_cast<void (ConfigMap::*)(const std::string&, std::string)>(
               &ConfigMap::edit),
           py::arg("key"), py::arg("value"))
      .def("exist_u64", &ConfigMap::existU64, py::arg("key"))
      .def("exist_i64", &ConfigMap::existI64, py::arg("key"))
      .def("exist_double", &ConfigMap::existDouble, py::arg("key"))
      .def("exist_str", &ConfigMap::existString, py::arg("key"))
      .def("exist", &ConfigMap::exist, py::arg("key"))
      .def("get_u64", &ConfigMap::getU64, py::arg("key"))
      .def("get_i64", &ConfigMap::getI64, py::arg("key"))
      .def("get_double", &ConfigMap::getDouble, py::arg("key"))
      .def("get_str", &ConfigMap::getString, py::arg("key"))
      .def("to_string", &ConfigMap::toString, py::arg("separator") = "\t",
           py::arg("new_line") = "\n")
      .def("from_string", &ConfigMap::fromString, py::arg("src"),
           py::arg("separator") = "\t", py::arg("new_line") = "\n")
      .def("clone_into", &ConfigMap::cloneInto, py::arg("dest"))
      .def("load_from", &ConfigMap::loadFrom, py::arg("src"))
      .def("to_file", &ConfigMap::toFile, py::arg("fname"),
           py::arg("separator") = ",", py::arg("new_line") = "\n")
      .def("from_file", &ConfigMap::fromFile, py::arg("fname"),
           py::arg("separator") = ",", py::arg("new_line") = "\n")
      .def(
          "from_carg",
          [](INTELLI::ConfigMap& self, int argc,
             std::vector<std::string> argv) {
            std::vector<const char*> c_args;
            for (const auto& str : argv) {
              c_args.push_back(str.c_str());
            }
            return self.fromCArg(argc, const_cast<char**>(c_args.data()));
          },
          py::arg("argc"), py::arg("argv"))
      .def("try_u64", &ConfigMap::tryU64, py::arg("key"),
           py::arg("default_value") = 0, py::arg("show_warning") = false)
      .def("try_i64", &ConfigMap::tryI64, py::arg("key"),
           py::arg("default_value") = 0, py::arg("show_warning") = false)
      .def("try_string", &ConfigMap::tryString, py::arg("key"),
           py::arg("default_value") = "", py::arg("show_warning") = false)
      .def("get_str_map", &ConfigMap::getStrMap)
      .def("get_i64_map", &ConfigMap::getI64Map)
      .def("get_double_map", &ConfigMap::getDoubleMap);
}