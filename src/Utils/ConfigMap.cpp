/*
 * Copyright (C) 2024/10/26 by the INTELLI team
 * Created on: 2024/10/26 13:14
 * Description: ${DESCRIPTION}
 */

#include <Utils/ConfigMap.hpp>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace INTELLI {
    ConfigMap::ConfigMap() = default;

    ConfigMap::~ConfigMap() = default;

    ConfigMap::ConfigMap(const ConfigMap &other) {
        loadConfig(other);
    }

    void ConfigMap::loadConfig(const ConfigMap &other) {
        u64Map = other.u64Map;
        i64Map = other.i64Map;
        doubleMap = other.doubleMap;
        strMap = other.strMap;
        conf = other.conf;
    }

    void ConfigMap::split(const std::string &s, const std::string &delimiter, std::vector<std::string> &v) {
        size_t start = 0;
        size_t end = s.find(delimiter);
        while (end != std::string::npos) {
            v.push_back(s.substr(start, end - start));
            start = end + delimiter.length();
            end = s.find(delimiter, start);
        }
        if (start < s.length()) {
            v.push_back(s.substr(start));
        }
    }

    /**
    * @brief Try to get an I64 from config map, if not exist, use default value instead
    * @param key The key
    * @param defaultValue The default
    * @param showWarning Whether show warning logs if not found
    * @return The returned value
    */
    int64_t ConfigMap::tryI64(const std::string &key, int64_t defaultValue = 0, bool showWarning = false) {
        int64_t ru = defaultValue;
        if (this->existI64(key)) {
            ru = this->getI64(key);
            // INTELLI_INFO(key + " = " + to_string(ru));
        } else {
            if (showWarning) {
                //INTELLI_WARNING("Leaving " + key + " as blank, will use " + to_string(defaultValue) + " instead");
            }
            //  WM_WARNNING("Leaving " + key + " as blank, will use " + to_string(defaultValue) + " instead");
        }
        return ru;
    }

    /**
   * @brief To detect whether the key exists and related to a I64
   * @param key
   * @return bool for the result
     */
    bool ConfigMap::existI64(const std::string &key) const {
        return (i64Map.count(key) == 1);
    }

    void ConfigMap::trim(std::string &s) {
        size_t start = s.find_first_not_of(" \t\n\r");
        if (start != std::string::npos) {
            s = s.substr(start);
        }
        size_t end = s.find_last_not_of(" \t\n\r");
        if (end != std::string::npos) {
            s = s.substr(0, end + 1);
        } else {
            s.clear();
        }
    }

    void ConfigMap::smartParse(const std::string &key, const std::string &value) {
        if ((std::isdigit(value[0]) || value[0] == '-' || value[0] == '+') && value.find('\'') == std::string::npos) {
            if (value.find('.') != std::string::npos) {
                double doubleValue = std::stod(value);
                edit(key, doubleValue);
            } else {
                int64_t intValue = std::stoll(value);
                edit(key, intValue);
            }
        } else if (value.front() == '\'' && value.back() == '\'') {
            edit(key, value.substr(1, value.length() - 2));
        } else {
            edit(key, value);
        }
    }


    void ConfigMap::edit(const std::string &key, uint64_t value) { u64Map[key] = value; }
    void ConfigMap::edit(const std::string &key, int64_t value) { i64Map[key] = value; }
    void ConfigMap::edit(const std::string &key, double value) { doubleMap[key] = value; }
    void ConfigMap::edit(const std::string &key, const std::string &value) { strMap[key] = value; }

    bool ConfigMap::exists(const std::string &key) const {
        return u64Map.count(key) || i64Map.count(key) || doubleMap.count(key) || strMap.count(key) || conf.count(key);
    }

    uint64_t ConfigMap::getU64(const std::string &key) const { return u64Map.at(key); }
    int64_t ConfigMap::getI64(const std::string &key) const { return i64Map.at(key); }
    double ConfigMap::getDouble(const std::string &key) const { return doubleMap.at(key); }
    std::string ConfigMap::getString(const std::string &key) const { return strMap.at(key); }

    std::string ConfigMap::getString(const std::string &key, const std::string &default_value) const {
        auto it = conf.find(key);
        if (it != conf.end()) {
            if (std::holds_alternative<std::string>(it->second)) {
                return std::get<std::string>(it->second);
            }
        }
        return default_value;
    }

    std::string ConfigMap::toString(const std::string &separator, const std::string &newLine) const {
        std::ostringstream oss;
        oss << "key" << separator << "value" << separator << "type" << newLine;
        for (const auto &[key, value]: u64Map) {
            oss << key << separator << value << separator << "U64" << newLine;
        }
        for (const auto &[key, value]: i64Map) {
            oss << key << separator << value << separator << "I64" << newLine;
        }
        for (const auto &[key, value]: doubleMap) {
            oss << key << separator << value << separator << "Double" << newLine;
        }
        for (const auto &[key, value]: strMap) {
            oss << key << separator << value << separator << "String" << newLine;
        }
        return oss.str();
    }

    bool ConfigMap::fromString(const std::string &src, const std::string &separator, const std::string &newLine) {
        std::istringstream ins(src);
        std::string readStr;
        while (std::getline(ins, readStr, newLine[0])) {
            std::vector<std::string> cols;
            split(readStr, separator, cols);
            if (cols.size() >= 3) {
                if (cols[2] == "U64") {
                    edit(cols[0], std::stoull(cols[1]));
                } else if (cols[2] == "I64") {
                    edit(cols[0], std::stoll(cols[1]));
                } else if (cols[2] == "Double") {
                    edit(cols[0], std::stod(cols[1]));
                } else if (cols[2] == "String") {
                    edit(cols[0], cols[1]);
                }
            }
        }
        return true;
    }

    bool ConfigMap::toFile(const std::string &fname, const std::string &separator, const std::string &newLine) const {
        std::ofstream of(fname);
        if (!of) {
            return false;
        }
        of << toString(separator, newLine);
        return true;
    }

    bool ConfigMap::fromFile(const std::string &fname, const std::string &separator, const std::string &newLine) {
        std::ifstream ins(fname);
        if (!ins) {
            return false;
        }
        std::ostringstream buffer;
        buffer << ins.rdbuf();
        return fromString(buffer.str(), separator, newLine);
    }

    int ConfigMap::parseIni(const std::string &fname) {
        std::ifstream file(fname);
        if (!file.is_open()) {
            throw std::runtime_error("Unable to open file: " + fname);
        }

        std::string line;
        while (std::getline(file, line)) {
            trim(line);

            if (line.empty() || line[0] == ';' || line[0] == '#') {
                continue;
            }

            size_t equal_pos = line.find('=');
            if (equal_pos != std::string::npos) {
                std::string key = line.substr(0, equal_pos);
                std::string value = line.substr(equal_pos + 1);
                trim(key);
                trim(value);

                try {
                    if (value.find('.') != std::string::npos) {
                        conf[key] = std::stof(value);
                    } else {
                        conf[key] = std::stoi(value);
                    }
                } catch (...) {
                    conf[key] = value;
                }
            }
        }
        return 0;
    }

    int ConfigMap::parseCsv(const std::string &fname) {
        std::ifstream file(fname);
        if (!file.is_open()) {
            throw std::runtime_error("Unable to open file: " + fname);
        }

        std::string line;
        while (std::getline(file, line)) {
            std::vector<std::string> cols;
            split(line, ",", cols);
            if (cols.size() >= 3) {
                std::istringstream iss(cols[1]);
                if (cols[2] == "U64" || cols[2] == "I64" || cols[2] == "Int") {
                    int value;
                    iss >> value;
                    conf[cols[0]] = value;
                } else if (cols[2] == "Double" || cols[2] == "Float") {
                    float value;
                    iss >> value;
                    conf[cols[0]] = value;
                } else if (cols[2] == "String") {
                    conf[cols[0]] = cols[1];
                }
            }
        }
        return 0;
    }


    int ConfigMap::getInt(const std::string &key, int default_value) const {
        auto it = conf.find(key);
        if (it != conf.end()) {
            if (std::holds_alternative<int>(it->second)) {
                return std::get<int>(it->second);
            }
        }
        return default_value;
    }

    float ConfigMap::getFloat(const std::string &key, float default_value) const {
        auto it = conf.find(key);
        if (it != conf.end()) {
            if (std::holds_alternative<float>(it->second)) {
                return std::get<float>(it->second);
            }
        }
        return default_value;
    }
} // namespace INTELLI
