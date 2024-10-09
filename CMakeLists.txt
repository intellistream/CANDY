# Copyright (C) 2021 by the INTELLI team (https://github.com/intellistream)

cmake_minimum_required(VERSION 3.14)
project(IntelliStream CXX)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Determine OS and enable Log4cxx if UNIX and not Apple
if (UNIX AND NOT APPLE)
    add_definitions(-DUSELOG4CXX)
    message(STATUS "Enable Log4CXX")
endif()

# Custom CMake find instructions and macros
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")
include(cmake/macros.cmake)
include(cmake/DefaultOptions.cmake)

# Check for g++ 11 or higher
find_program(GPP_COMPILER NAMES g++)

if(NOT GPP_COMPILER)
    message(FATAL_ERROR "g++ compiler is required. Please install g++ 11 or higher.")
else()
    execute_process(COMMAND ${GPP_COMPILER} --version OUTPUT_VARIABLE GPP_VERSION_OUTPUT OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" GPP_VERSION ${GPP_VERSION_OUTPUT})
    string(REGEX MATCH "^([0-9]+)" GPP_VERSION_MAJOR ${GPP_VERSION})

    if(GPP_VERSION_MAJOR LESS 11)
        message(STATUS "g++ version found: ${GPP_VERSION}")
        message(FATAL_ERROR "g++ 11 or higher is required. Please install it and try again.")
    else()
        message(STATUS "g++ version ${GPP_VERSION} found")
    endif()
endif()

# Set Compilation Flags
set(CMAKE_CXX_FLAGS "-std=c++20 -Wall -Werror=return-type -Werror=unused-variable -Werror=unused-parameter")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DNO_RACE_CHECK -DIntelliStream_DEBUG_MODE=1")
set(CMAKE_CXX_FLAGS_RELEASE "-Wno-ignored-qualifiers -Wno-sign-compare -O3")

# Set Logging Level Flag
if (IntelliStream_LOGGING_LEVEL)
    get_log_level_value(IntelliStream_LOGGING_VALUE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DIntelliStream_LOGGING_LEVEL=${IntelliStream_LOGGING_VALUE}")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DIntelliStream_LOGGING_LEVEL=${IntelliStream_LOGGING_VALUE}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DIntelliStream_LOGGING_LEVEL=${IntelliStream_LOGGING_VALUE}")
else()
    message(STATUS "Logging everything (no specific log level set).")
endif()

message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")

# Log4cxx setup
find_package(Log4cxx REQUIRED)
include_directories(${Log4cxx_INCLUDE_DIR})
set(LIBRARIES ${LIBRARIES} ${Log4cxx_LIBRARY})

if (LOG4CXX_FOUND)
    message(STATUS "LOG4CXX found")
else()
    message(FATAL_ERROR "LOG4CXX is required but not found. Please install it (e.g., sudo apt-get install -y liblog4cxx-dev) and try again.")
endif()

# Enable Unit Tests if requested
option(ENABLE_UNIT_TESTS "Enable unit tests" ON)
message(STATUS "Enable testing: ${ENABLE_UNIT_TESTS}")

if (ENABLE_UNIT_TESTS)
    enable_testing()
    include(FetchContent)

    FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG release-1.11.0
    )

    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    include(GoogleTest)
endif()

# Print all used include directories
message(STATUS "INCLUDE_DIRS:")
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach (dir ${dirs})
    message(STATUS "  - ${dir}")
endforeach()

message(STATUS "Libraries: ${LIBRARIES}")

# Add Source Code
add_subdirectory(src)

# Add Library
get_sources(IntelliStream_SOURCE_FILES)
get_headers(IntelliStream_HEADER_FILES)
add_library(IntelliStream SHARED ${IntelliStream_SOURCE_FILES} ${IntelliStream_HEADER_FILES})
target_include_directories(IntelliStream PUBLIC "include")
target_include_directories(IntelliStream PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(IntelliStream PUBLIC ${LIBRARIES})

# Add applications to test the effectivness of CANDY. For a more comprehensive benchmark evaluation, please checkout https://github.com/intellistream/CANDY-Benchmark
add_subdirectory(apps)

# Add tests (based on GTest)
add_subdirectory(test)

# Install Headers
install(DIRECTORY "include" DESTINATION "/IntelliStream" COMPONENT IntelliStream)