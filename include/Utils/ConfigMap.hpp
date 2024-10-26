/*
 * Copyright (C) 2024/10/26 by the INTELLI team
 * Created on: 2024/10/26 13:14
 * Description: ${DESCRIPTION}
 */
#ifndef _UTILS_CONFIGMAP_HPP_
#define _UTILS_CONFIGMAP_HPP_

#include <map>
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>

namespace INTELLI {
 /**
  * @class ConfigMap
  * @brief The unified map structure to store configurations in a key-value style
  */
 class ConfigMap {
 protected:
  std::map<std::string, uint64_t> u64Map;
  std::map<std::string, int64_t> i64Map;
  std::map<std::string, double> doubleMap;
  std::map<std::string, std::string> strMap;
  std::unordered_map<std::string, std::variant<int, float, std::string> > conf;

  static void split(const std::string &s, const std::string &delimiter, std::vector<std::string> &v);

  static void trim(std::string &s);

  void smartParse(const std::string &key, const std::string &value);

 public:
  ConfigMap();

  ~ConfigMap();

  ConfigMap(const ConfigMap &other);

  void loadConfig(const ConfigMap &other);

  void edit(const std::string &key, uint64_t value);

  void edit(const std::string &key, int64_t value);

  void edit(const std::string &key, double value);

  void edit(const std::string &key, const std::string &value);

  template<typename T>
  void edit(const std::string &key, const T &value);

  bool exists(const std::string &key) const;

  template<typename T>
  T get(const std::string &key) const;

  uint64_t getU64(const std::string &key) const;

  int64_t getI64(const std::string &key) const;

  double getDouble(const std::string &key) const;

  std::string getString(const std::string &key) const;

  std::string toString(const std::string &separator = "\t", const std::string &newLine = "\n") const;

  bool fromString(const std::string &src, const std::string &separator = "\t", const std::string &newLine = "\n");

  bool toFile(const std::string &fname, const std::string &separator = ",", const std::string &newLine = "\n") const;

  bool fromFile(const std::string &fname, const std::string &separator = ",", const std::string &newLine = "\n");

  int parseIni(const std::string &fname);

  int parseCsv(const std::string &fname);

  std::string getString(const std::string &key, const std::string &default_value) const;

  int getInt(const std::string &key, int default_value = 0) const;

  float getFloat(const std::string &key, float default_value = 0.0f) const;
 };

 using ConfigMapPtr = std::shared_ptr<ConfigMap>;
#define newConfigMap std::make_shared<INTELLI::ConfigMap>
} // namespace INTELLI

#endif // _UTILS_CONFIGMAP_HPP_
