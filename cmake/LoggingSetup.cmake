# cmake/LoggingSetup.cmake

# Set Logging Level Flag
if (IntelliStream_LOGGING_LEVEL)
    get_log_level_value(IntelliStream_LOGGING_VALUE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DIntelliStream_LOGGING_LEVEL=${IntelliStream_LOGGING_VALUE}")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DIntelliStream_LOGGING_LEVEL=${IntelliStream_LOGGING_VALUE}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DIntelliStream_LOGGING_LEVEL=${IntelliStream_LOGGING_VALUE}")
else ()
    message(STATUS "Logging everything (no specific log level set).")
endif ()
