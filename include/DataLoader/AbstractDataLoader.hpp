/*
* Copyright (C) 2024 by the INTELLI team
 * Created on: 2024/10/9
 * Description: [Provide description here]
 */
#ifndef CANDY_INCLUDE_DATALOADER_AbstractDataLoader_H_
#define CANDY_INCLUDE_DATALOADER_AbstractDataLoader_H_

#include <Utils/ConfigMap.hpp>
#include <Utils/TensorOP.hpp>
#include <memory>

namespace CANDY_ALGO {

class AbstractDataLoader {
 public:
  AbstractDataLoader() = default;
  virtual ~AbstractDataLoader() = default;

  virtual bool setConfig(INTELLI::ConfigMapPtr cfg);

  /**
   * @brief get the data tensor
   * @return the generated data tensor
   */
  virtual torch::Tensor getData();

  /**
    * @brief get the data tensor at specific offset
    * @note implement and use this when the whole data tensor does not fit into main memory
    * @return the generated data tensor
    */
  virtual torch::Tensor getDataAt(int64_t startPos, int64_t endPos);

  /**
  * @brief get the query tensor
  * @return the generated query tensor
  */
  virtual torch::Tensor getQuery();

  /**
   * @brief get the number of rows of data
   * @return the rows
   */
  virtual int64_t size();
};

typedef std::shared_ptr<class CANDY_ALGO::AbstractDataLoader>
    AbstractDataLoaderPtr;

#define newAbstractDataLoader std::make_shared<CANDY_ALGO::AbstractDataLoader>

}  // namespace CANDY_ALGO

#endif  //CANDY_INCLUDE_MATRIXLOADER_AbstractDataLoader_H_
