//
// Created by tony on 24-11-4.
//
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <Utils/ConfigMap.hpp>
#include <Algorithms/FlatGPUIndex/FlatGPUIndex.hpp>
#include <Algorithms/KNN/KNNSearch.hpp>
#include <Algorithms/ANNSBase.hpp>
#include <string>
#include <DataLoader/DataLoaderTable.hpp>
namespace py = pybind11;
using namespace INTELLI;
using namespace CANDY_ALGO;
py::dict configMapToDict(const std::shared_ptr<INTELLI::ConfigMap> &cfg) {
  py::dict d;
  auto i64Map = cfg->getI64Map();
  auto doubleMap = cfg->getDoubleMap();
  auto strMap = cfg->getStrMap();
  for (auto &iter : i64Map) {
    d[py::cast(iter.first)] = py::cast(iter.second);
  }
  for (auto &iter : doubleMap) {
    d[py::cast(iter.first)] = py::cast(iter.second);
  }
  for (auto &iter : strMap) {
    d[py::cast(iter.first)] = py::cast(iter.second);
  }
  return d;
}

// Function to convert Python dictionary to ConfigMap
std::shared_ptr<INTELLI::ConfigMap> dictToConfigMap(const py::dict &dict) {
  auto cfg = std::make_shared<INTELLI::ConfigMap>();
  for (auto item : dict) {
    auto key = py::str(item.first);
    auto value = item.second;
    // Check if the type is int
    if (py::isinstance<py::int_>(value)) {
      int64_t val = value.cast<int64_t>();
      cfg->edit(key, val);
      //std::cout << "Key: " << key.cast<std::string>() << " has an int value." << std::endl;
    }
      // Check if the type is float
    else if (py::isinstance<py::float_>(value)) {
      double val = value.cast<float>();
      cfg->edit(key, val);
    }
      // Check if the type is string
    else if (py::isinstance<py::str>(value)) {
      std::string val = py::str(value);
      cfg->edit(key, val);
    }
      // Add more type checks as needed
    else {
      std::cout << "Key: " << key.cast<std::string>() << " has a value of another type." << std::endl;
    }
  }
  return cfg;
}

ANNSBasePtr createIndex(std::string nameTag) {
  std::map<std::string, CANDY_ALGO::ANNSBasePtr> indexMap;
  indexMap["flat"] = newKNNIndex();
  indexMap["knn"] = newKNNIndex();
  indexMap["flatGPU"] =  newFlatGPUIndex();
  return indexMap[nameTag];
}

AbstractDataLoaderPtr creatDataLoader(std::string nameTag) {
  DataLoaderTable dt;
  auto ru = dt.findDataLoader(nameTag);
  if (ru == nullptr) {
    INTELLI_ERROR("No index named " + nameTag + ", return flat");
    nameTag = "random";
    return dt.findDataLoader(nameTag);
  }
  return ru;
}

// Define the compile time as a string using the __DATE__ and __TIME__ macros
#define COMPILED_TIME (__DATE__ " " __TIME__)
PYBIND11_MODULE(PyCANDYAlgos, m) {
  m.attr("__version__") = "0.0.1";  // Set the version of the module
  m.attr("__compiled_time__") = COMPILED_TIME;  // Set the compile time of the module
  m.def("configMapToDict", &configMapToDict, "A function that converts ConfigMap to Python dictionary");
  m.def("dictToConfigMap", &dictToConfigMap, "A function that converts  Python dictionary to ConfigMap");
  py::class_<INTELLI::ConfigMap, std::shared_ptr<INTELLI::ConfigMap>>(m, "ConfigMap")
      .def(py::init<>())
      .def("edit", py::overload_cast<const std::string &, int64_t>(&INTELLI::ConfigMap::edit))
      .def("edit", py::overload_cast<const std::string &, double>(&INTELLI::ConfigMap::edit))
      .def("edit", py::overload_cast<const std::string &, std::string>(&INTELLI::ConfigMap::edit))
      .def("toString", &INTELLI::ConfigMap::toString,
           py::arg("separator") = "\t",
           py::arg("newLine") = "\n")
      .def("toFile", &ConfigMap::toFile,
           py::arg("fname"),
           py::arg("separator") = ",",
           py::arg("newLine") = "\n")
      .def("fromFile", &ConfigMap::fromFile,
           py::arg("fname"),
           py::arg("separator") = ",",
           py::arg("newLine") = "\n");
  m.def("createIndex", &createIndex, "A function to create new index by name tag");
  py::class_<ANNSBase, std::shared_ptr<ANNSBase>>(m, "ANNSBase")
      .def(py::init<>())
      .def("reset", &ANNSBase::reset, py::call_guard<py::gil_scoped_release>())
      .def("setConfig", &ANNSBase::setConfig, py::call_guard<py::gil_scoped_release>())
      .def("startHPC", &ANNSBase::startHPC)
      .def("insertTensor", &ANNSBase::insertTensor)
      .def("loadInitialTensor", &ANNSBase::loadInitialTensor)
      .def("deleteTensor", &ANNSBase::deleteTensor)
      .def("reviseTensor", &ANNSBase::reviseTensor)
      .def("searchTensor", &ANNSBase::searchTensor)
      .def("endHPC", &ANNSBase::endHPC)
      .def("resetIndexStatistics", &ANNSBase::resetIndexStatistics)
      .def("getIndexStatistics", &ANNSBase::getIndexStatistics);
  m.def("createDataLoader", &creatDataLoader, "A function to create new data loader by name tag");
  py::class_<CANDY_ALGO::AbstractDataLoader, std::shared_ptr<CANDY_ALGO::AbstractDataLoader>>(m, "AbstractDataLoader")
      .def(py::init<>())
      .def("setConfig", &CANDY_ALGO::AbstractDataLoader::setConfig)
      .def("getData", &CANDY_ALGO::AbstractDataLoader::getData)
      .def("getDataAt", &CANDY_ALGO::AbstractDataLoader::getDataAt)
      .def("getQuery", &CANDY_ALGO::AbstractDataLoader::getQuery);
}