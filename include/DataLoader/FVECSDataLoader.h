/*
 * Copyright (C) 2024 by the INTELLI team
 * Created on: 24-10-30 上午10:50
 * Description: ${DESCRIPTION}
 */
#ifndef CANDY_INCLUDE_DataLOADER_FVECSDATALOADER_H
#define CANDY_INCLUDE_DataLOADER_FVECSDATALOADER_H

#include <Utils/ConfigMap.hpp>
#include <Utils/TensorOP.hpp>
#include <memory>
#include <DataLoader/AbstractDataLoader.hpp>

namespace CANDY {

    class FVECSDataLoader : public AbstractDataLoader {
        protected:
            torch::Tensor A, B;
            int64_t vecDim, vecVolume, querySize, seed;
            int64_t normalizeTensor;
            double queryNoiseFraction;
            int64_t useSeparateQuery;
            bool generateData(std::string fname);
            bool generateQuery(std::string fname);

        public:
            FVECSDataLoader() = default;

            ~FVECSDataLoader() = default;

            /**
               * @brief Set the GLOBAL config map related to this loader
               * @param cfg The config map
                * @return bool whether the config is successfully set
                * @note
               */
            virtual bool setConfig(INTELLI::ConfigMapPtr cfg);

            /**
             * @brief get the data tensor
             * @return the generated data tensor
             */
            virtual torch::Tensor getData();

            /**
            * @brief get the query tensor
            * @return the generated query tensor
            */
            virtual torch::Tensor getQuery();

            /**
             * @brief the inline function to load tensor from fvecs file
             * @param fname the name of file
             * @return the genearetd tensor
             */
            static torch::Tensor tensorFromFVECS(std::string fname);
        };

    typedef std::shared_ptr<class CANDY::FVECSDataLoader> FVECSDataLoaderPtr;

#define newFVECSDataLoader std::make_shared<CANDY::FVECSDataLoader>
}

#endif //CANDY_INCLUDE_DataLOADER_FVECSDATALOADER_H
