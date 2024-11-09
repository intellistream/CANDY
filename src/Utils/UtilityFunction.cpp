/*
 * Copyright (C) 2024 by the INTELLI team
 * Created on: 24-11-6 下午8:06
 * Description: ${DESCRIPTION}
 */

#include <Utils/UtilityFunctions.hpp>
#include <sched.h>
#include <pthread.h>
#include <cstdlib>



int INTELLI::UtilityFunctions::bind2Core(int id) {
  if (id == -1) //OS scheduling
  {
    return -1;
  }
  int maxCpu = std::thread::hardware_concurrency();
  int cpuId = id % maxCpu;
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(cpuId, &mask);
  /**
   * @brief fixed some core bind bugs
   */
  if (sched_setaffinity(0, sizeof(cpu_set_t), &mask) < 0) {
    printf("Error: setaffinity()\n");
    exit(0);
  }
  return cpuId;
}

double INTELLI::UtilityFunctions::getLatencyPercentage(double fraction, std::vector<INTELLI::IntelliTimeStampPtr> &myTs) {
  size_t rLen = myTs.size();
  size_t nonZeroCnt = 0;
  std::vector<uint64_t> validLatency;
  for (size_t i = 0; i < rLen; i++) {
    if (myTs[i]->processedTime >= myTs[i]->arrivalTime && myTs[i]->processedTime != 0) {
      validLatency.push_back(myTs[i]->processedTime - myTs[i]->arrivalTime);
      nonZeroCnt++;
    }
  }
  if (nonZeroCnt == 0) {
    INTELLI_ERROR("No valid latency, maybe there is no AMM result?");
    return 0;
  }
  std::sort(validLatency.begin(), validLatency.end());
  double t = nonZeroCnt;
  t = t * fraction;
  size_t idx = (size_t) t + 1;
  if (idx >= validLatency.size()) {
    idx = validLatency.size() - 1;
  }
  return validLatency[idx];
}

bool INTELLI::UtilityFunctions::saveTimeStampToFile(std::string fname,
                              std::vector<INTELLI::IntelliTimeStampPtr> &myTs,
                              bool skipZero) {
  ofstream of;
  of.open(fname);
  if (of.fail()) {
    return false;
  }
  of << "eventTime,arrivalTime,processedTime\n";
  size_t rLen = myTs.size();
  for (size_t i = 0; i < rLen; i++) {
    if (skipZero && myTs[i]->processedTime == 0) {

    } else {
      auto tp = myTs[i];
      string line = to_string(tp->eventTime) + ","
          + to_string(tp->arrivalTime) + "," + to_string(tp->processedTime) + "\n";
      of << line;
    }

  }
  of.close();
  return true;
}

bool INTELLI::UtilityFunctions::existRow(torch::Tensor base, torch::Tensor row) {
  for (int64_t i = 0; i < base.size(0); i++) {
    auto tensor1 = base[i].contiguous();
    auto tensor2 = row.contiguous();
    if (torch::equal(tensor1, tensor2)) {
      return true;
    }
  }
  return false;
}

double INTELLI::UtilityFunctions::calculateRecall(std::vector<torch::Tensor> groundTruth, std::vector<torch::Tensor> prob) {
  int64_t truePositives = 0;
  int64_t falseNegatives = 0;
  for (size_t i = 0; i < prob.size(); i++) {
    auto gdI = groundTruth[i];
    auto probI = prob[i];
    for (int64_t j = 0; j < probI.size(0); j++) {
      if (existRow(gdI, probI[j])) {
        truePositives++;
      } else {
        falseNegatives++;
      }
    }
  }
  double recall = static_cast<double>(truePositives) / (truePositives + falseNegatives);
  return recall;
}

bool INTELLI::UtilityFunctions::tensorListToFile(std::vector<torch::Tensor> &tensorVec, std::string folderName) {
  try {
    std::filesystem::remove_all(folderName);
  } catch (const std::filesystem::filesystem_error &e) {
  }

  try {
    // Create the folder
    std::filesystem::create_directory(folderName);
  } catch (const std::filesystem::filesystem_error &e) {
  }

  for (size_t i = 0; i < tensorVec.size(); i++) {
    std::string fileName = folderName + "/" + std::to_string(i) + ".rbt";
    TensorOP::tensorToFile(&tensorVec[i], fileName);
  }
  return true;
}

std::vector<torch::Tensor> INTELLI::UtilityFunctions::tensorListFromFile(std::string folderName, uint64_t tensors) {
  std::vector<torch::Tensor> ru((size_t) tensors);
  for (uint64_t i = 0; i < tensors; i++) {
    std::string fileName = folderName + "/" + std::to_string(i) + ".rbt";
    TensorOP::tensorFromFile(&ru[i], fileName);
  }
  return ru;
}
