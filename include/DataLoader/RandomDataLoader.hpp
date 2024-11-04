/*
* Copyright (C) 2024 by the INTELLI team
* Created on: 2024/10/9
* Description: [Provide description here]
*/
#ifndef CANDY_INCLUDE_DATALOADER_RandomDataLoader_H_
#define CANDY_INCLUDE_DATALOADER_RandomDataLoader_H_

#include <Utils/ConfigMap.hpp>
#include <Utils/TensorOP.hpp>
#include <memory>
#include <DataLoader/AbstractDataLoader.hpp>

namespace CANDY_ALGO {

 class RandomDataLoader : public AbstractDataLoader {
 protected:
  torch::Tensor A, B;
  int64_t vecDim, vecVolume, querySize, seed;
  int64_t driftPosition;
  double driftOffset, queryNoiseFraction;

 public:
  RandomDataLoader() = default;

  ~RandomDataLoader() = default;

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
 };

 /**
  * @ingroup CANDY_MatrixLOADER_Random
  * @typedef RandomDataLoaderPtr
  * @brief The class to describe a shared pointer to @ref RandomDataLoader

  */
 typedef std::shared_ptr<class CANDY_ALGO::RandomDataLoader> RandomDataLoaderPtr;
 /**
  * @ingroup CANDY_MatrixLOADER_Random
  * @def newRandomDataLoader
  * @brief (Macro) To creat a new @ref RandomDataLoader under shared pointer.
  */
#define newRandomDataLoader std::make_shared<CANDY_ALGO::RandomDataLoader>
} // CANDY

#endif //CANDY_INCLUDE_MATRIXLOADER_RandomDataLoader_H_
