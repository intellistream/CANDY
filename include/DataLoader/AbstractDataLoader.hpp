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

namespace CANDY {

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

 typedef std::shared_ptr<class CANDY::AbstractDataLoader> AbstractDataLoaderPtr;

#define newAbstractDataLoader std::make_shared<CANDY::AbstractDataLoader>

} // CANDY

#endif //CANDY_INCLUDE_MATRIXLOADER_AbstractDataLoader_H_
