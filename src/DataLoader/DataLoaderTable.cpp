/*
* Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/9
 * Description: [Provide description here]
 */
#include <DataLoader/FVECSDataLoader.h>
#include <DataLoader/DataLoaderTable.hpp>
#include <DataLoader/RandomDataLoader.hpp>

// #include <include/hdf5_config.h>
// #if CANDY_HDF5 == 1
// #include <DataLoader/HDF5DataLoader.h>
// #endif

namespace CANDY_ALGO {
/**
     * @note revise me if you need new loader
     */
CANDY_ALGO::DataLoaderTable::DataLoaderTable() {
  loaderMap["null"] = newAbstractDataLoader();
  loaderMap["random"] = newRandomDataLoader();
  loaderMap["fvecs"] = newFVECSDataLoader();
  // #if CANDY_HDF5 == 1
  // loaderMap["hdf5"] = newHDF5DataLoader();
  // #endif
}
}  // namespace CANDY_ALGO