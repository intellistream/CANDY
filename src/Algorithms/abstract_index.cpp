/*
* Copyright (C) 2024 by the INTELLI team
 * Created by: Ziao Wang
 * Created on: 2024/10/22
 * Description: [Provide description here]
 */

#include <Algorithms/abstract_index.hpp>
#include <cassert>

static std::vector<std::string> u64ObjectToStringObject(
    std::vector<uint64_t>& u64s) {
  std::vector<std::string> ru(u64s.size());
  for (size_t i = 0; i < u64s.size(); i++) {
    uint64_t u64i = u64s[i];
    const char* char_ptr = reinterpret_cast<const char*>(&u64i);
    ru[i] = std::string(char_ptr, sizeof(uint64_t));
  }
  return ru;
}

static std::vector<uint64_t> stringObjectToU64Object(
    std::vector<std::string>& strs) {
  std::vector<uint64_t> ru(strs.size());
  for (size_t i = 0; i < strs.size(); i++) {
    uint64_t u64i = 0;
    std::memcpy(&u64i, strs[i].data(), sizeof(uint64_t));
    ru[i] = u64i;
  }
  return ru;
}

void AbstractIndex::reset() {}

bool AbstractIndex::offlineBuild(torch::Tensor& t) {
  return false;
}

bool AbstractIndex::setConfig(ConfigParserPtr cfg) {
  assert(cfg);
  std::string metricType = cfg->get_string("metricType", "L2");
  Metric = METRIC_L2;
  if (metricType == "dot" || metricType == "IP" || metricType == "cossim") {
    Metric = METRIC_INNER_PRODUCT;
  }
  return true;
}

bool AbstractIndex::setConfigClass(ConfigParser cfg) {
  ConfigParserPtr cfgPtr = newConfigParser(cfg);
  return setConfig(cfgPtr);
}

bool AbstractIndex::setFrozenLevel(int64_t frozenLv) {
  assert(frozenLv >= 0);
  return false;
}

bool AbstractIndex::insertTensor(torch::Tensor& t) {
  assert(t.size(1));
  return false;
}

bool AbstractIndex::insertStringObject(torch::Tensor& t,
                                       std::vector<std::string>& strs) {
  assert(t.size(1));
  assert(strs.size());
  return false;
}

bool AbstractIndex::insertU64Object(torch::Tensor& t,
                                    std::vector<uint64_t>& u64s) {
  auto strVec = u64ObjectToStringObject(u64s);
  return insertStringObject(t, strVec);
}

bool AbstractIndex::loadInitialU64Object(torch::Tensor& t,
                                         std::vector<uint64_t>& u64s) {
  auto strVec = u64ObjectToStringObject(u64s);
  return loadInitialStringObject(t, strVec);
}

bool AbstractIndex::loadInitialTensor(torch::Tensor& t) {
  return insertTensor(t);
}

bool AbstractIndex::loadInitialStringObject(torch::Tensor& t,
                                            std::vector<std::string>& strs) {
  return insertStringObject(t, strs);
}

bool AbstractIndex::deleteTensor(torch::Tensor& t, int64_t k) {
  assert(t.size(1));
  assert(k > 0);
  return false;
}

bool AbstractIndex::deleteU64Object(torch::Tensor& t, int64_t k) {
  return deleteStringObject(t, k);
}

bool AbstractIndex::deleteStringObject(torch::Tensor& t, int64_t k) {
  assert(t.size(1));
  assert(k > 0);
  return false;
}

bool AbstractIndex::reviseTensor(torch::Tensor& t, torch::Tensor& w) {
  assert(t.size(1) == w.size(1));
  return false;
}

std::vector<idx_t> AbstractIndex::searchIndex(torch::Tensor q, int64_t k) {
  assert(k > 0);
  assert(q.size(1));
  std::vector<idx_t> ru(1);
  return ru;
}

std::vector<std::vector<std::string>> AbstractIndex::searchStringObject(
    torch::Tensor& q, int64_t k) {
  assert(k > 0);
  assert(q.size(1));
  std::vector<std::vector<std::string>> ru(1);
  ru[0] = std::vector<std::string>(1);
  ru[0][0] = "";
  return ru;
}

std::vector<std::vector<uint64_t>> AbstractIndex::searchU64Object(
    torch::Tensor& q, int64_t k) {
  auto ruS = searchStringObject(q, k);
  std::vector<std::vector<uint64_t>> ruU =
      std::vector<std::vector<uint64_t>>(ruS.size());
  for (size_t i = 0; i < ruU.size(); i++) {
    ruU[i] = stringObjectToU64Object(ruS[i]);
  }
  return ruU;
}

std::vector<torch::Tensor> AbstractIndex::getTensorByIndex(
    std::vector<idx_t>& idx, int64_t k) {
  assert(k > 0);
  assert(idx.size());
  std::vector<torch::Tensor> ru(1);
  ru[0] = torch::rand({1, 1});
  return ru;
}

torch::Tensor AbstractIndex::rawData() {
  return torch::rand({1, 1});
}

std::vector<torch::Tensor> AbstractIndex::searchTensor(torch::Tensor& q,
                                                       int64_t k) {
  assert(k > 0);
  assert(q.size(1));
  std::vector<torch::Tensor> ru(1);
  ru[0] = torch::rand({1, 1});
  return ru;
}

bool AbstractIndex::startHPC() {
  return false;
}

bool AbstractIndex::endHPC() {
  return false;
}

bool AbstractIndex::waitPendingOperations() {
  return true;
}

std::tuple<std::vector<torch::Tensor>, std::vector<std::vector<std::string>>>
AbstractIndex::searchTensorAndStringObject(torch::Tensor& q, int64_t k) {
  auto ruT = searchTensor(q, k);
  auto ruS = searchStringObject(q, k);
  std::tuple<std::vector<torch::Tensor>, std::vector<std::vector<std::string>>>
      ru(ruT, ruS);
  return ru;
}

bool AbstractIndex::loadInitialTensorAndQueryDistribution(
    torch::Tensor& t, torch::Tensor& query) {
  assert(query.size(0) > 0);
  return loadInitialTensor(t);
}

bool AbstractIndex::resetIndexStatistics() {
  return false;
}

ConfigParserPtr AbstractIndex::getIndexStatistics() {
  auto ru = newConfigParser();
  ru->edit("hasExtraStatistics", 0);
  return ru;
}
