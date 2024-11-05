/*
* Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/9
 * Description: [Provide description here]
 */
#ifndef CANDY_INCLUDE_DataLOADER_DataLOADERTABLE_H_
#define CANDY_INCLUDE_DataLOADER_DataLOADERTABLE_H_

#include <DataLoader/AbstractDataLoader.hpp>
#include <map>

namespace CANDY_ALGO {

#define newDataLoaderTable std::make_shared<CANDY_ALGO::DataLoaderTable>

class DataLoaderTable {
 protected:
  std::map<std::string, CANDY_ALGO::AbstractDataLoaderPtr> loaderMap;

 public:
  DataLoaderTable();

  ~DataLoaderTable() {}

  void registerNewDataLoader(CANDY_ALGO::AbstractDataLoaderPtr dnew,
                             std::string tag) {
    loaderMap[tag] = dnew;
  }

  CANDY_ALGO::AbstractDataLoaderPtr findDataLoader(std::string name) {

    if (loaderMap.count(name)) {
      return loaderMap[name];
    }
    return nullptr;
  }

  typedef std::shared_ptr<class CANDY_ALGO::DataLoaderTable> DataLoaderTablePtr;
};

}  // namespace CANDY_ALGO

#endif  //INTELLISTREAM_INCLUDE_DataLOADER_DataLOADERTABLE_H_
