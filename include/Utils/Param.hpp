/*
 * Copyright (C) 2024/10/26 by the INTELLI team
 * Created on: 2024/10/26 21:35
 * Description: ${DESCRIPTION}
 */

#ifndef PARAM_HPP
#define PARAM_HPP
#include <cstdint>
#include <memory>

namespace CANDY {
class Param;
typedef std::shared_ptr<Param> ParamPtr;

class Param {
 public:
  // for kdtree
  int64_t num_trees;
};
}  // namespace CANDY

#endif  //PARAM_HPP
