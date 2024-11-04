/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/9
 * Description: [Provide description here]
 */
#ifndef INTELLISTREAM_SRC_UTILS_LOGGING_HPP_
#define INTELLISTREAM_SRC_UTILS_LOGGING_HPP_

#include <iostream>
#include <string>
#include <stdexcept>
#include <sstream>
#include <c10/util/Logging.h>
enum DebugLevel { LOG_NONE, LOG_WARNING, LOG_DEBUG, LOG_INFO, LOG_TRACE };

static std::string getDebugLevelAsString(DebugLevel level) {
  switch (level) {
    case LOG_NONE: return "LOG_NONE";
    case LOG_WARNING: return "LOG_WARNING";
    case LOG_DEBUG: return "LOG_DEBUG";
    case LOG_INFO: return "LOG_INFO";
    case LOG_TRACE: return "LOG_TRACE";
    default: return "UNKNOWN";
  }
}

static DebugLevel getStringAsDebugLevel(const std::string& level) {
  if (level == "LOG_NONE") {
    return LOG_NONE;
  } else if (level == "LOG_WARNING") {
    return LOG_WARNING;
  } else if (level == "LOG_DEBUG") {
    return LOG_DEBUG;
  } else if (level == "LOG_INFO") {
    return LOG_INFO;
  } else if (level == "LOG_TRACE") {
    return LOG_TRACE;
  } else {
    throw std::runtime_error("Logger: Debug level unknown: " + level);
  }
}

#define INTELLI_TRACE(TEXT)                                                                                                          \
    do {                                                                                                                             \
        std::cout << "TRACE: " << TEXT << std::endl;                                                                               \
    } while (0);

#define INTELLI_DEBUG(TEXT)                                                                                                          \
    do {                                                                                                                             \
        std::cout << "DEBUG: " << TEXT << std::endl;                                                                               \
    } while (0);

#define INTELLI_INFO(TEXT)                                                                                                           \
    do {                                                                                                                             \
        std::cout << "INFO: " << TEXT << std::endl;                                                                                \
    } while (0);

#define INTELLI_WARNING(TEXT)                                                                                                        \
    do {                                                                                                                             \
        std::cout << "WARNING: " << TEXT << std::endl;                                                                             \
    } while (0);

#define INTELLI_ERROR(TEXT)                                                                                                          \
    do {                                                                                                                             \
        std::cerr << "ERROR: " << TEXT << std::endl;                                                                               \
    } while (0);

#define INTELLI_FATAL_ERROR(TEXT)                                                                                                    \
    do {                                                                                                                             \
        std::cerr << "FATAL: " << TEXT << std::endl;                                                                               \
        throw std::runtime_error(TEXT);                                                                                              \
    } while (0);

#define INTELLI_ASSERT(CONDITION, TEXT)                                                                                              \
    do {                                                                                                                             \
        if (!(CONDITION)) {                                                                                                          \
            std::cerr << "ASSERTION FAILED: " << TEXT << std::endl;                                                                \
            throw std::runtime_error("INTELLI Runtime Error on condition " #CONDITION);                                            \
        }                                                                                                                            \
    } while (0);

static void setupLogging(const std::string& logFileName, DebugLevel level) {
  std::cout << "LogFileName: " << logFileName << ", and DebugLevel: " << getDebugLevelAsString(level) << std::endl;
}

#define C10_INFO(n) LOG(INFO)<<n

#define C10_ERROR(n) LOG(ERROR)<<n
#define C10_WARNING(n) LOG(WARNING)<<n

#endif //INTELLISTREAM_SRC_UTILS_LOGGING_HPP_
