/*
* Copyright (C) 2024 by the INTELLI team
 * Created by: Jianjun Zhao
 * Created on: 2024/12/21
 * Description:
 */

#ifndef CANDY_INCLUDE_STORAGE_AbstractStorageEngine_H_
#define CANDY_INCLUDE_STORAGE_AbstractStorageEngine_H_
#include <torch/torch.h>
#include <ComputeEngine/AbstractComputeEngine.hpp>
#include <string>
#include <vector>
namespace CANDY_STORAGE{
class AbstractStorageEngine {
public:
    AbstractStorageEngine() = default;
    virtual ~AbstractStorageEngine() = default;
    CANDY_COMPUTE::AbstractComputeEnginePtr compute_engine;
    /**
     * @brief Get the index-specific to one vector
     * @return vid
     */
    virtual int getVid() = 0;
    /**
     * @brief Insert a tensor
     * @param vector the tensor to be inserted
     * @return bool whether the insertion is successful
     */
    virtual bool insertTensor(const torch::Tensor &vector) = 0;
    /**
     * @brief Insert a tensor
     * @param vector the tensor to be inserted
     * @param vid the vid of the inserted vector to be returned 
     * @return bool whether the insertion is successful
     */
    virtual bool insertTensor(const torch::Tensor &vector, int64_t &vid) = 0;
    /**
     * @brief Delete a tensor
     * @param vids the vids of the tensors to be deleted
     * @return vectors
     */
    virtual std::vector<torch::Tensor> deleteTensor(std::vector<int64_t> vids) = 0;
    /**
     * @brief Revise a tensor
     * @param r_t the tensor to be revised
     * @param vid the vid of the tensor to be revised
     * @return bool whether the revision is successful
     */
    virtual bool reviseTensor(const torch::Tensor &r_t, int64_t vid) = 0;
    /**
     * @brief Compute the distance between two vectors
     * @param vid1 the vid of the first vector
     * @param vid2 the vid of the second vector
     * @return float the distance between the two vectors
     */
    virtual float distanceCompute(int64_t vid1, int64_t vid2) = 0;
   /**
     * @brief Compute the distance between a vector and a tensor
     * @param vector the vector
     * @param vid the vid of the tensor
     * @return float the distance between the vector and the tensor
     */
    virtual float distanceCompute(const torch::Tensor &vector, int64_t vid) = 0;
    /**
     * @brief Get the vector by vid
     * @param vid the vid of the vector
     * @return torch::Tensor the vector
     */
    virtual torch::Tensor getVectorByVid(int64_t vid) = 0;
     /**
    * @brief Get the vectors by vids
    * @param vid the vids of the vectors
    * @return torch::Tensor the vectors
    */
   virtual std::vector<torch::Tensor> getVectorByVids(std::vector<int64_t> vids) = 0;
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
typedef std::shared_ptr<AbstractStorageEngine> AbstractStorageEnginePtr;
}

#endif  //CANDY_INCLUDE_STORAGE_AbstractStorageEngine_H_