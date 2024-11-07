/*
* Copyright (C) 2024 by the INTELLI team
 * Created on: 2024/11/6
 * Description: [Provide description here]
 */

#ifndef IntelliStream_SRC_UTILS_UTILITYFUNCTIONS_HPP_
#define IntelliStream_SRC_UTILS_UTILITYFUNCTIONS_HPP_

#include <barrier>
#include <sys/time.h>
#include <experimental/filesystem>
#include <functional>
#include <string>
//#include <torch/torch.h>
//#include <ATen/ATen.h>
//#include <Common/Types.h>
#include <torch/torch.h>
#include <filesystem>
#include <vector>
#include <Utils/TensorOP.hpp>
#include <Utils/TimeStampGenerator.hpp>



namespace INTELLI {



#define TIME_LAST_UNIT_MS 1000
#define TIME_LAST_UNIT_US 1000000
#define chronoElapsedTime(start)                         \
  std::chrono::duration_cast<std::chrono::microseconds>( \
      std::chrono::high_resolution_clock::now() - start) \
      .count()

/**
  * UtilityFunctions` notes
  *    calculateRecall(); Computes the recall metric.
  *    ↳  existRow(); Checks if a specific row from one tensor exists in another tensor.
  *
  *    tensorListFromFile() Loads tensors from binary files.
  *    tensorListToFile(); Save tensors to binary files.
  *
  *    saveTimeStampToFile(); Records each timestamp’s eventTime, arrivalTime, and processedTime in a specified file.
  *
  *    getLatencyPercentage(); Calculates and returns the specified percentile latency.
  */


class UtilityFunctions {

 public:

  UtilityFunctions();

  //bind to CPU
  /*!
   bind to CPU
   \li bind the thread to core according to id
   \param id the core you plan to bind, -1 means let os decide
   \return cpuId, the real core that bind to
   \partition
    */
  static int bind2Core(int id);

  static double getLatencyPercentage(double fraction, std::vector<INTELLI::IntelliTimeStampPtr> &myTs);

  static bool saveTimeStampToFile(std::string fname,
                                std::vector<INTELLI::IntelliTimeStampPtr> &myTs,
                                bool skipZero = true);

  static bool existRow(torch::Tensor base, torch::Tensor row);

  static double calculateRecall(std::vector<torch::Tensor> groundTruth, std::vector<torch::Tensor> prob);

  static bool tensorListToFile(std::vector<torch::Tensor> &tensorVec, std::string folderName);

  static std::vector<torch::Tensor> tensorListFromFile(std::string folderName, uint64_t tensors);

};

}  // namespace INTELLI
#endif  //IntelliStream_SRC_UTILS_UTILITYFUNCTIONS_HPP_