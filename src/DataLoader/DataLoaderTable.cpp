/*
* Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/9
 * Description: [Provide description here]
 */
#include <DataLoader/DataLoaderTable.hpp>
#include <DataLoader/RandomDataLoader.hpp>

namespace CANDY_ALGO {
    /**
     * @note revise me if you need new loader
     */
    CANDY_ALGO::DataLoaderTable::DataLoaderTable() {
        loaderMap["null"] = newAbstractDataLoader();
        loaderMap["random"] = newRandomDataLoader();
    }
} // CANDY
