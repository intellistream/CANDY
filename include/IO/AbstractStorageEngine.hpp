/*
* Copyright (C) 2024 by the jjzhao
 * Created on: 2024/11/24
 * Description: [Provide description here]
 */

#ifndef CANDY_INCLUDE_STORAGE_AbstractStorageEngine_H_
#define CANDY_INCLUDE_STORAGE_AbstractStorageEngine_H_
#include <torch/torch.h>
#include <vector>
#include <string>

class AbstractStorageEngine {
  public:
    AbstractStorageEngine() = default;
    virtual ~AbstractStorageEngine() = default;
    /**
     * @brief Get the index-specific to one vector
     * @return vid
     */
    virtual int getVid() = 0;
    /**
     * @brief Insert a tensor with a raw id
     * @param vector the tensor to be inserted
     * @param rawId the raw id of the tensor
     * @return bool whether the insertion is successful
     */
    virtual bool insertTensorWithRawId(const torch::Tensor &vector, int rawId) = 0;
    /**
     * @brief Insert a tensor
     * @param vector the tensor to be inserted
     * @return bool whether the insertion is successful
     */
    virtual bool insertTensor(const torch::Tensor &vector) = 0;
    /**
     * @brief Delete a tensor
     * @param vids the vids of the tensors to be deleted
     * @return vector<int> the rowIds of the deleted tensors
     */
    virtual std::vector<int> deleteTensor(std::vector<int> vids) = 0;
    /**
     * @brief Compute the distance between two vectors
     * @param vid1 the vid of the first vector
     * @param vid2 the vid of the second vector
     * @return float the distance between the two vectors
     */
    virtual float distanceCompute(int vid1, int vid2) = 0;
   /**
     * @brief Compute the distance between a vector and a tensor
     * @param vector the vector
     * @param vid the vid of the tensor
     * @return float the distance between the vector and the tensor
     */
    virtual float distanceCompute(const torch::Tensor &vector, int vid) = 0;
    /**
     * @brief Get the vector by vid
     * @param vid the vid of the vector
     * @return torch::Tensor the vector
     */
    virtual torch::Tensor getVectorByVid(int vid) = 0;
    /**
     * @brief Get the rawId by vid
     * @param vid the vid of the vector
     * @return torch::Tensor the rowId
     */
    virtual int getRawIdByVid(int vid) = 0;
    /**
     * @brief Get all the vectors
     * @return std::vector<torch::Tensor> the vectors
     */
    virtual std::vector<torch::Tensor> getAll() = 0;
    /**
     * @brief Display the storage
     * @return string the display string
     */
    virtual std::string display() = 0;
};

#endif  //CANDY_INCLUDE_STORAGE_AbstractStorageEngine_H_