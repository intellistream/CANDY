// Copyright (C) 2021 by the INTELLI team (https://github.com/INTELLI)

#ifndef INTELLI_INCLUDE_UTILS_LOGGER_HPP_
#define INTELLI_INCLUDE_UTILS_LOGGER_HPP_
// TRACE < DEBUG < INFO < WARN < ERROR < FATAL
#include <iostream>
#ifdef USELOG4CXX
#include <log4cxx/consoleappender.h>
#include <log4cxx/fileappender.h>
#include <log4cxx/logger.h>
#include <log4cxx/patternlayout.h>

using namespace log4cxx;
using namespace log4cxx::helpers;
#endif

enum DebugLevel { LOG_NONE, LOG_WARNING, LOG_DEBUG, LOG_INFO, LOG_TRACE };

static std::string getDebugLevelAsString(DebugLevel level) {
  if (level == LOG_NONE) {
    return "LOG_NONE";
  } else if (level == LOG_WARNING) {
    return "LOG_WARNING";
  } else if (level == LOG_DEBUG) {
    return "LOG_DEBUG";
  } else if (level == LOG_INFO) {
    return "LOG_INFO";
  } else if (level == LOG_TRACE) {
    return "LOG_TRACE";
  } else {
    return "UNKNOWN";
  }
}

static DebugLevel getStringAsDebugLevel(std::string level) {
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
#ifdef USELOG4CXX
static log4cxx::LoggerPtr INTELLILogger(log4cxx::Logger::getLogger("INTELLI"));
#endif

// TRACE < DEBUG < INFO < WARN < ERROR < FATAL
#define LEVEL_TRACE 6
#define LEVEL_DEBUG 5
#define LEVEL_INFO 4
#define LEVEL_WARN 3
#define LEVEL_ERROR 2
#define LEVEL_FATAL 1

#ifdef INTELLI_LOGGING_LEVEL
#if INTELLI_LOGGING_LEVEL >= LEVEL_TRACE
#define INTELLI_TRACE(TEXT)                                                                                                          \
    do {                                                                                                                         \
        LOG4CXX_TRACE(INTELLILogger, TEXT);                                                                                          \
    } while (0)
#else
#define INTELLI_TRACE(TEXT)                                                                                                          \
    do {                                                                                                                         \
        std::ostringstream oss;                                                                                                  \
        ((void) (oss << TEXT));                                                                                                  \
    } while (0)
#endif

#if INTELLI_LOGGING_LEVEL >= LEVEL_DEBUG
#define INTELLI_DEBUG(TEXT)                                                                                                          \
    do {                                                                                                                         \
        LOG4CXX_DEBUG(INTELLILogger, TEXT);                                                                                          \
    } while (0)
#else
#define INTELLI_DEBUG(TEXT)                                                                                                          \
    do {                                                                                                                         \
        std::ostringstream oss;                                                                                                  \
        ((void) (oss << TEXT));                                                                                                  \
    } while (0)
#endif

#if INTELLI_LOGGING_LEVEL >= LEVEL_INFO
#define INTELLI_INFO(TEXT)                                                                                                           \
    do {                                                                                                                         \
        LOG4CXX_INFO(INTELLILogger, TEXT);                                                                                           \
    } while (0)
#else
#define INTELLI_INFO(TEXT)                                                                                                           \
    do {                                                                                                                         \
        std::ostringstream oss;                                                                                                  \
        ((void) (oss << TEXT));                                                                                                  \
    } while (0)
#endif

#if INTELLI_LOGGING_LEVEL >= LEVEL_WARN
#define INTELLI_WARNING(TEXT)                                                                                                        \
    do {                                                                                                                         \
        LOG4CXX_WARN(INTELLILogger, TEXT);                                                                                           \
    } while (0)
#else
#define INTELLI_WARNING(TEXT)                                                                                                        \
    do {                                                                                                                         \
        std::ostringstream oss;                                                                                                  \
        ((void) (oss << TEXT));                                                                                                  \
    } while (0)
#endif

#if INTELLI_LOGGING_LEVEL >= LEVEL_ERROR
#define INTELLI_ERROR(TEXT)                                                                                                          \
    do {                                                                                                                         \
        LOG4CXX_ERROR(INTELLILogger, TEXT);                                                                                          \
    } while (0)
#else
#define INTELLI_ERROR(TEXT)                                                                                                          \
    do {                                                                                                                         \
        std::ostringstream oss;                                                                                                  \
        ((void) (oss << TEXT));                                                                                                  \
    } while (0)
#endif

#if INTELLI_LOGGING_LEVEL >= LEVEL_FATAL
#define INTELLI_FATAL_ERROR(TEXT)                                                                                                    \
    do {                                                                                                                         \
        LOG4CXX_ERROR(INTELLILogger, TEXT);                                                                                          \
    } while (0)
#else
#define INTELLI_FATAL_ERROR(TEXT)                                                                                                    \
    do {                                                                                                                         \
        std::ostringstream oss;                                                                                                  \
        ((void) (oss << TEXT));                                                                                                  \
    } while (0)
#endif
#else
  #ifdef USELOG4CXX
    #define INTELLI_TRACE(TEXT)                                                                                                          \
        do {                                                                                                                            \
            LOG4CXX_TRACE(INTELLILogger, TEXT);                                                                                          \
        } while (0)
    #define INTELLI_DEBUG(TEXT)                                                                                                          \
        do {                                                                                                                         \
            LOG4CXX_DEBUG(INTELLILogger, TEXT);                                                                                          \
        } while (0)
    #define INTELLI_TRACE(TEXT)                                                                                                          \
        do {                                                                                                                         \
            LOG4CXX_TRACE(INTELLILogger, TEXT);                                                                                          \
        } while (0)
    #define INTELLI_INFO(TEXT)                                                                                                           \
        do {                                                                                                                         \
            LOG4CXX_INFO(INTELLILogger, TEXT);                                                                                           \
        } while (0)
    #define INTELLI_WARNING(TEXT)                                                                                                        \
        do {                                                                                                                         \
            LOG4CXX_WARN(INTELLILogger, TEXT);                                                                                           \
        } while (0)
    #define INTELLI_ERROR(TEXT)                                                                                                          \
        do {                                                                                                                         \
            LOG4CXX_ERROR(INTELLILogger, TEXT);                                                                                          \
        } while (0)
    #define INTELLI_FATAL_ERROR(TEXT)                                                                                                    \
        do {                                                                                                                         \
            LOG4CXX_ERROR(INTELLILogger, TEXT);                                                                                          \
        } while (0)
                                                                                               \
  #endif
  #ifndef USELOG4CXX
    #define INTELLI_TRACE(TEXT)                                                                                                          \
      std::cout << TEXT << std::endl;
    #define INTELLI_DEBUG(TEXT)                                                                                                          \
      std::cout << TEXT << std::endl;
    #define INTELLI_TRACE(TEXT)                                                                                                          \
      std::cout << TEXT << std::endl;
    #define INTELLI_INFO(TEXT)                                                                                                           \
      std::cout << TEXT << std::endl;
    #define INTELLI_WARNING(TEXT)                                                                                                        \
      std::cout << TEXT << std::endl;
    #define INTELLI_ERROR(TEXT)                                                                                                          \
      std::cout << TEXT << std::endl;
    #define INTELLI_FATAL_ERROR(TEXT)                                                                                                    \
      std::cout << TEXT << std::endl;

  #endif
#endif

#ifdef INTELLI_DEBUG
#define INTELLI_VERIFY(CONDITION, TEXT)                                                                                              \
    do {                                                                                                                         \
        if (!(CONDITION)) {                                                                                                      \
            INTELLI::collectAndPrintStacktrace();                                                                                    \
            INTELLI_FATAL_ERROR(TEXT);                                                                                               \
            throw std::runtime_error("INTELLI Runtime Error on condition " #CONDITION);                                              \
        }                                                                                                                        \
    } while (0)
#else
#define INTELLI_VERIFY(CONDITION, TEXT) ((void) 0)
#endif

#define INTELLI_ASSERT(CONDITION, TEXT)                                                                                              \
    do {                                                                                                                         \
        if (!(CONDITION)) {                                                                                                      \
            INTELLI::collectAndPrintStacktrace();                                                                                    \
            INTELLI_FATAL_ERROR(TEXT);                                                                                               \
            throw std::runtime_error("INTELLI Runtime Error on condition " #CONDITION);                                              \
        }                                                                                                                        \
    } while (0)

#define INTELLI_ASSERT2(CONDITION, ...)                                                                                              \
    do {                                                                                                                         \
        if (!(CONDITION)) {                                                                                                      \
            INTELLI::collectAndPrintStacktrace();                                                                                    \
            INTELLI_FATAL_ERROR(__VA_ARGS__);                                                                                        \
            throw std::runtime_error("INTELLI Runtime Error on condition " #CONDITION);                                              \
        }                                                                                                                        \
    } while (0)

#define INTELLI_THROW_RUNTIME_ERROR(TEXT)                                                                                            \
    do {                                                                                                                         \
        INTELLI::collectAndPrintStacktrace();                                                                                        \
        INTELLI_FATAL_ERROR(TEXT);                                                                                                   \
        throw std::runtime_error(TEXT);                                                                                          \
    } while (0)

  static void setupLogging(std::string logFileName, DebugLevel level) {
      std::cout << "LogFileName: " << logFileName << ", and DebugLevel: " << level << std::endl;
      #ifdef USELOG4CXX
      std::cout << "Logger: SETUP_LOGGING" << std::endl;
      // create PatternLayout
      log4cxx::LayoutPtr
          layoutPtr(new log4cxx::PatternLayout("%d{MMM dd yyyy HH:mm:ss} %c: %l %X{threadName} [%-5t] [%p] : %m%n"));

      // create FileAppender
      LOG4CXX_DECODE_CHAR(fileName, logFileName);
      log4cxx::FileAppenderPtr file(new log4cxx::FileAppender(layoutPtr, fileName));

      // create ConsoleAppender
      log4cxx::ConsoleAppenderPtr console(new log4cxx::ConsoleAppender(layoutPtr));

        // set log level
      #ifdef INTELLI_LOGGING_LEVEL
          ((void) level);
        #if INTELLI_LOGGING_LEVEL == LEVEL_FATAL
            INTELLILogger->setLevel(log4cxx::Level::getFatal());
        #endif
        #if INTELLI_LOGGING_LEVEL == LEVEL_ERROR
            INTELLILogger->setLevel(log4cxx::Level::getError());
        #endif
        #if INTELLI_LOGGING_LEVEL == LEVEL_WARN
            INTELLILogger->setLevel(log4cxx::Level::getWarn());
        #endif
        #if INTELLI_LOGGING_LEVEL == LEVEL_INFO
            INTELLILogger->setLevel(log4cxx::Level::getInfo());
        #endif
        #if INTELLI_LOGGING_LEVEL == LEVEL_DEBUG
            INTELLILogger->setLevel(log4cxx::Level::getDebug());
        #endif
        #if INTELLI_LOGGING_LEVEL == LEVEL_TRACE
            INTELLILogger->setLevel(log4cxx::Level::getTrace());
        #endif
      #else
        if (level == LOG_NONE) {
          INTELLILogger->setLevel(log4cxx::Level::getOff());
        } else if (level == LOG_WARNING) {
          INTELLILogger->setLevel(log4cxx::Level::getWarn());
        } else if (level == LOG_DEBUG) {
          INTELLILogger->setLevel(log4cxx::Level::getDebug());
        } else if (level == LOG_INFO) {
          INTELLILogger->setLevel(log4cxx::Level::getInfo());
        } else if (level == LOG_TRACE) {
          INTELLILogger->setLevel(log4cxx::Level::getTrace());
        } else {
          INTELLI_ERROR("setupLogging: log level not supported " << getDebugLevelAsString(level));
          throw Exception("Error while setup logging");
        }
      #endif

        INTELLILogger->addAppender(file);
        INTELLILogger->addAppender(console);
      }

      static void setLogLevel(DebugLevel level) {
        // set log level
      #ifdef INTELLI_LOGGING_LEVEL
          ((void) level);
        #if INTELLI_LOGGING_LEVEL == LEVEL_FATAL
            INTELLILogger->setLevel(log4cxx::Level::getFatal());
        #endif
        #if INTELLI_LOGGING_LEVEL == LEVEL_ERROR
            INTELLILogger->setLevel(log4cxx::Level::getError());
        #endif
        #if INTELLI_LOGGING_LEVEL == LEVEL_WARN
            INTELLILogger->setLevel(log4cxx::Level::getWarn());
        #endif
        #if INTELLI_LOGGING_LEVEL == LEVEL_INFO
            INTELLILogger->setLevel(log4cxx::Level::getInfo());
        #endif
        #if INTELLI_LOGGING_LEVEL == LEVEL_DEBUG
            INTELLILogger->setLevel(log4cxx::Level::getDebug());
        #endif
        #if INTELLI_LOGGING_LEVEL == LEVEL_TRACE
            INTELLILogger->setLevel(log4cxx::Level::getTrace());
        #endif
      #else

        if (level == LOG_NONE) {
          INTELLILogger->setLevel(log4cxx::Level::getOff());
        } else if (level == LOG_WARNING) {
          INTELLILogger->setLevel(log4cxx::Level::getWarn());
        } else if (level == LOG_DEBUG) {
          INTELLILogger->setLevel(log4cxx::Level::getDebug());
        } else if (level == LOG_INFO) {
          INTELLILogger->setLevel(log4cxx::Level::getInfo());
        } else if (level == LOG_TRACE) {
          INTELLILogger->setLevel(log4cxx::Level::getTrace());
        } else {
          INTELLI_ERROR("setLogLevel: log level not supported " << getDebugLevelAsString(level));
          throw Exception("Error while trying to change log level");
        }
      #endif
    #endif
  }


#define INTELLI_NOT_IMPLEMENTED()                                                                                                    \
    do {                                                                                                                         \
        INTELLI_ERROR("Function Not Implemented!");                                                                                  \
        throw Exception("not implemented");                                                                                      \
    } while (0)

#endif //INTELLI_INCLUDE_UTILS_LOGGER_HPP_
