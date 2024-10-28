/*
* Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/9
 * Description: [Provide description here]
 */
#ifndef CANDY_INCLUDE_DataLOADER_DataLOADERTABLE_H_
#define CANDY_INCLUDE_DataLOADER_DataLOADERTABLE_H_

#include <map>
#include <DataLoader/AbstractDataLoader.hpp>

namespace CANDY {

#define newDataLoaderTable std::make_shared<CANDY::DataLoaderTable>

class DataLoaderTable {
 protected:
  std::map<std::string, CANDY::AbstractDataLoaderPtr> loaderMap;
 public:
  DataLoaderTable();
  ~DataLoaderTable() {
  }
  void registerNewDataLoader(CANDY::AbstractDataLoaderPtr dnew, std::string tag) {
    loaderMap[tag] = dnew;
  }
  CANDY::AbstractDataLoaderPtr findDataLoader(std::string name) {
    if (loaderMap.count(name)) {
      return loaderMap[name];
    }
    return nullptr;
  }

  typedef std::shared_ptr<class CANDY::DataLoaderTable> DataLoaderTablePtr;


};

} // CANDY

#endif //INTELLISTREAM_INCLUDE_DataLOADER_DataLOADERTABLE_H_
