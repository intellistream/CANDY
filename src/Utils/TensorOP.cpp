/* TensorOP.cpp */

#include <Utils/TensorOP.hpp>
#include <algorithm>
#include <fstream>

namespace INTELLI {

bool TensorOP::deleteRow(torch::Tensor* tensor, int64_t rowIdx) {
  if (rowIdx >= tensor->size(0))
    return false;
  auto rowMask = torch::arange(tensor->size(0)).to(torch::kLong).ne(rowIdx);
  *tensor = tensor->index({rowMask.nonzero().squeeze()});
  return true;
}

bool TensorOP::deleteRow(TensorPtr tensor, int64_t rowIdx) {
  return deleteRow(tensor.get(), rowIdx);
}

bool TensorOP::deleteRows(torch::Tensor* tensor, std::vector<int64_t>& rowIdx) {
  auto rowMask = torch::ones({tensor->size(0)}).to(torch::kLong);
  for (int64_t row : rowIdx) {
    if (row >= tensor->size(0))
      return false;
    rowMask[row] = 0;
  }
  *tensor = tensor->index({rowMask.nonzero().squeeze()});
  return true;
}

bool TensorOP::deleteRows(TensorPtr tensor, std::vector<int64_t>& rowIdx) {
  return deleteRows(tensor.get(), rowIdx);
}

bool TensorOP::appendRows(torch::Tensor* tHead, torch::Tensor* tTail) {
  if (tHead->size(1) != tTail->size(1))
    return false;
  *tHead = torch::cat({*tHead, *tTail}, 0);
  return true;
}

bool TensorOP::appendRows(TensorPtr tHead, TensorPtr tTail) {
  return appendRows(tHead.get(), tTail.get());
}

bool TensorOP::insertRows(torch::Tensor* tHead, torch::Tensor* tTail,
                          int64_t startRow) {
  if (tHead->size(1) != tTail->size(1))
    return false;
  auto topPart =
      tHead->index({torch::indexing::Slice(torch::indexing::None, startRow)});
  auto bottomPart =
      tHead->index({torch::indexing::Slice(startRow, torch::indexing::None)});
  *tHead = torch::cat({topPart, *tTail, bottomPart}, 0);
  return true;
}

bool TensorOP::insertRows(TensorPtr tHead, TensorPtr tTail, int64_t startRow) {
  return insertRows(tHead.get(), tTail.get(), startRow);
}

bool TensorOP::editRows(torch::Tensor* tHead, const torch::Tensor* tTail,
                        int64_t startRow) {
  if (tHead->size(1) != tTail->size(1))
    return false;
  int64_t endRow = startRow + tTail->size(0);
  if (endRow > tHead->size(0)) {
    tHead->slice(0, startRow, tHead->size(0)) =
        tTail->slice(0, 0, tHead->size(0) - startRow);
  } else {
    tHead->slice(0, startRow, endRow) = *tTail;
  }
  return true;
}

bool TensorOP::editRows(TensorPtr tHead, TensorPtr tTail, int64_t startRow) {
  return editRows(tHead.get(), tTail.get(), startRow);
}

bool TensorOP::deleteRowBufferMode(torch::Tensor* tensor, int64_t rowIdx,
                                   int64_t* lastNNZ) {
  if (rowIdx >= tensor->size(0) || *lastNNZ >= tensor->size(0) ||
      rowIdx > *lastNNZ)
    return false;
  tensor->slice(0, rowIdx, rowIdx + 1) =
      tensor->slice(0, *lastNNZ, *lastNNZ + 1);
  tensor->slice(0, *lastNNZ, *lastNNZ + 1) = torch::zeros({1, tensor->size(1)});
  if (*lastNNZ > 0)
    *lastNNZ -= 1;
  return true;
}

bool TensorOP::deleteRowBufferMode(TensorPtr tensor, int64_t rowIdx,
                                   int64_t* lastNNZ) {
  return deleteRowBufferMode(tensor.get(), rowIdx, lastNNZ);
}

bool TensorOP::deleteRowsBufferMode(torch::Tensor* tensor,
                                    std::vector<int64_t>& rowIdx,
                                    int64_t* lastNNZ) {
  std::sort(rowIdx.begin(), rowIdx.end(), std::greater<int64_t>());
  for (int64_t value : rowIdx) {
    if (!deleteRowBufferMode(tensor, value, lastNNZ))
      return false;
  }
  return true;
}

bool TensorOP::deleteRowsBufferMode(TensorPtr tensor,
                                    std::vector<int64_t>& rowIdx,
                                    int64_t* lastNNZ) {
  return deleteRowsBufferMode(tensor.get(), rowIdx, lastNNZ);
}

bool TensorOP::appendRowsBufferMode(torch::Tensor* tHead,
                                    const torch::Tensor* tTail,
                                    int64_t* lastNNZ,
                                    int64_t customExpandSize) {
  if (tHead->size(1) != tTail->size(1))
    return false;
  if (*lastNNZ + tTail->size(0) < tHead->size(0)) {
    if (editRows(tHead, tTail, *lastNNZ + 1)) {
      *lastNNZ += tTail->size(0);
      return true;
    }
  } else {
    int64_t requiredExpandSize = *lastNNZ + tTail->size(0) + 1 - tHead->size(0);
    int64_t expandSize = std::max(requiredExpandSize, customExpandSize);
    *tHead =
        torch::cat({*tHead, torch::zeros({expandSize, tHead->size(1)})}, 0);
    if (editRows(tHead, tTail, *lastNNZ + 1)) {
      *lastNNZ += tTail->size(0);
      return true;
    }
  }
  return false;
}

bool TensorOP::appendRowsBufferMode(TensorPtr tHead, TensorPtr tTail,
                                    int64_t* lastNNZ,
                                    int64_t customExpandSize) {
  return appendRowsBufferMode(tHead.get(), tTail.get(), lastNNZ,
                              customExpandSize);
}

std::vector<uint8_t> TensorOP::tensorToFlatBin(torch::Tensor* tensor) {
  auto A_size = tensor->sizes();
  int64_t rows = A_size[0];
  int64_t cols = A_size[1];
  uint64_t packedSize = tensor->numel() * sizeof(float) + sizeof(int64_t) * 2;
  std::vector<uint8_t> ru(packedSize);
  auto ruIter = ru.begin();
  std::copy(reinterpret_cast<const uint8_t*>(&rows),
            reinterpret_cast<const uint8_t*>(&rows) + sizeof(int64_t), ruIter);
  ruIter += sizeof(int64_t);
  std::copy(reinterpret_cast<const uint8_t*>(&cols),
            reinterpret_cast<const uint8_t*>(&cols) + sizeof(int64_t), ruIter);
  ruIter += sizeof(int64_t);
  std::copy(reinterpret_cast<const uint8_t*>(tensor->data_ptr<float>()),
            reinterpret_cast<const uint8_t*>(tensor->data_ptr<float>() +
                                             tensor->numel()),
            ruIter);
  return ru;
}

bool TensorOP::tensorToFile(torch::Tensor* tensor, const std::string& fname) {
  std::ofstream file(fname, std::ios::binary);
  if (!file.is_open())
    return false;
  auto vec = tensorToFlatBin(tensor);
  file.write(reinterpret_cast<const char*>(vec.data()), vec.size());
  return file.good();
}

bool TensorOP::tensorFromFlatBin(torch::Tensor* tensor,
                                 std::vector<uint8_t>& ru) {
  if (ru.size() < sizeof(int64_t) * 2)
    return false;
  int64_t rows, cols;
  std::copy(ru.begin(), ru.begin() + sizeof(int64_t),
            reinterpret_cast<uint8_t*>(&rows));
  std::copy(ru.begin() + sizeof(int64_t), ru.begin() + 2 * sizeof(int64_t),
            reinterpret_cast<uint8_t*>(&cols));
  uint64_t expectedSize = rows * cols * sizeof(float) + sizeof(int64_t) * 2;
  if (ru.size() < expectedSize)
    return false;
  *tensor = torch::from_blob(ru.data() + 2 * sizeof(int64_t), {rows, cols},
                             torch::kFloat32)
                .clone();
  return true;
}

bool TensorOP::tensorFromFile(torch::Tensor* tensor, const std::string& fname) {
  std::ifstream file(fname, std::ios::binary);
  if (!file.is_open())
    return false;
  file.seekg(0, std::ios::end);
  std::streamsize fileSize = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<uint8_t> binaryData(fileSize);
  file.read(reinterpret_cast<char*>(binaryData.data()), fileSize);
  return tensorFromFlatBin(tensor, binaryData);
}

torch::Tensor TensorOP::rowSampling(torch::Tensor& tensor,
                                    int64_t sampledRows) {
  if (sampledRows >= tensor.size(0) || sampledRows <= 0)
    return tensor.clone();
  auto indices =
      torch::randperm(tensor.size(0), torch::kLong).slice(0, 0, sampledRows);
  return tensor.index_select(0, indices);
}

torch::Tensor TensorOP::l2Normalize(torch::Tensor& tensor) {
  torch::Tensor norm = torch::norm(tensor, 2, 0, true);
  return tensor / norm;
}
}  // namespace INTELLI
