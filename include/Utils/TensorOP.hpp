/* TensorOP.hpp */
#ifndef _UTILS_INTELLI_TENSOR_OP_H_
#define _UTILS_INTELLI_TENSOR_OP_H_

#pragma once
#include <vector>
#include <torch/torch.h>
#include <memory>

namespace INTELLI {

typedef std::shared_ptr<torch::Tensor> TensorPtr;
#define newTensor std::make_shared<torch::Tensor>

class TensorOP {
 public:
  TensorOP() = default;
  ~TensorOP() = default;

  static bool deleteRow(torch::Tensor *tensor, int64_t rowIdx);
  static bool deleteRow(TensorPtr tensor, int64_t rowIdx);
  static bool deleteRows(torch::Tensor *tensor, std::vector<int64_t> &rowIdx);
  static bool deleteRows(TensorPtr tensor, std::vector<int64_t> &rowIdx);
  static bool appendRows(torch::Tensor *tHead, torch::Tensor *tTail);
  static bool appendRows(TensorPtr tHead, TensorPtr tTail);
  static bool insertRows(torch::Tensor *tHead, torch::Tensor *tTail, int64_t startRow);
  static bool insertRows(TensorPtr tHead, TensorPtr tTail, int64_t startRow);
  static bool editRows(torch::Tensor *tHead, torch::Tensor *tTail, int64_t startRow);
  static bool editRows(TensorPtr tHead, TensorPtr tTail, int64_t startRow);
  static bool deleteRowBufferMode(torch::Tensor *tensor, int64_t rowIdx, int64_t *lastNNZ);
  static bool deleteRowBufferMode(TensorPtr tensor, int64_t rowIdx, int64_t *lastNNZ);
  static bool deleteRowsBufferMode(torch::Tensor *tensor, std::vector<int64_t> &rowIdx, int64_t *lastNNZ);
  static bool deleteRowsBufferMode(TensorPtr tensor, std::vector<int64_t> &rowIdx, int64_t *lastNNZ);
  static bool appendRowsBufferMode(torch::Tensor *tHead, torch::Tensor *tTail, int64_t *lastNNZ, int64_t customExpandSize = 0);
  static bool appendRowsBufferMode(TensorPtr tHead, TensorPtr tTail, int64_t *lastNNZ, int64_t customExpandSize = 0);
  static std::vector<uint8_t> tensorToFlatBin(torch::Tensor *tensor);
  static bool tensorToFile(torch::Tensor *tensor, const std::string &fname);
  static bool tensorFromFlatBin(torch::Tensor *tensor, std::vector<uint8_t> &ru);
  static bool tensorFromFile(torch::Tensor *tensor, const std::string &fname);
  static torch::Tensor rowSampling(torch::Tensor &tensor, int64_t sampledRows);
  static torch::Tensor l2Normalize(torch::Tensor &tensor);
};

} // namespace INTELLI

#endif // _UTILS_INTELLI_TENSOR_OP_H_