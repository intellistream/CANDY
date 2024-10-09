# Copyright (C) 2021 by the INTELLI team (https://github.com/intellistream)

cmake_minimum_required(VERSION 3.14)
project(IntelliStream CXX)

# Judge the os system

if (UNIX AND NOT APPLE)
    add_definitions(-DUSELOG4CXX)
    MESSAGE(STATUS "Enable Log4CXX")
endif (UNIX AND NOT APPLE)

# Custom CMake find instructions and macros
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")
include(cmake/macros.cmake)
include(cmake/default.cmake)

# C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#gcc 10 g++10
message(STATUS "sudo add-apt-repository 'deb http://mirrors.kernel.org/ubuntu hirsute main universe'")
message(STATUS "sudo apt-get update")
message(STATUS "sudo apt install gcc-11 g++-11")
message(STATUS "sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11")
message(STATUS "sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 11")

# Set Optimization Flags
set(CMAKE_CXX_FLAGS "-std=c++20 -Wall -Werror=return-type -Werror=unused-variable -Werror=unused-parameter")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DNO_RACE_CHECK -DIntelliStream_DEBUG_MODE=1")
set(CMAKE_CXX_FLAGS_RELEASE "-Wno-ignored-qualifiers -Wno-sign-compare -O3")

# Set LOGGING_LEVEL Flag
if (IntelliStream_LOGGING_LEVEL)
    get_log_level_value(IntelliStream_LOGGING_VALUE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DIntelliStream_LOGGING_LEVEL=${IntelliStream_LOGGING_VALUE}")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DIntelliStream_LOGGING_LEVEL=${IntelliStream_LOGGING_VALUE}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DIntelliStream_LOGGING_LEVEL=${IntelliStream_LOGGING_VALUE}")
else (IntelliStream_LOGGING_LEVEL)
    message("---Everything will be logged")
endif (IntelliStream_LOGGING_LEVEL)

message(STATUS "CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE}")


# Log4cc
find_package(Log4cxx REQUIRED)
include_directories(${Log4cxx_INCLUDE_DIR})
set(LIBRARIES ${LIBRARIES} ${Log4cxx_LIBRARY})

if (LOG4CXX_FOUND)
    message(STATUS "LOG4CXX found")
else (LOG4CXX_FOUND)
    message(STATUS "LOG4CXX not found")
    message(STATUS "On Ubuntu, install the package by the following command!")
    message(STATUS "sudo apt-get install -y liblog4cxx-dev")
endif (LOG4CXX_FOUND)

option(ENABLE_UNIT_TESTS "Enable unit tests" ON)
message(STATUS "Enable testing: ${ENABLE_UNIT_TESTS}")

if (ENABLE_UNIT_TESTS)
    enable_testing()
    # Google Test
    include(FetchContent)

    FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG release-1.11.0
    )

    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    include(GoogleTest)
endif ()

# Print all used include directories
message(STATUS "INCLUDE_DIRS:")
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach (dir ${dirs})
    message(STATUS "  - ${dir}")
endforeach ()

message(STATUS " Libraries: ${LIBRARIES}")

# Add Source Code
add_subdirectory(src)

# Add Library
get_sources(IntelliStream_SOURCE_FILES)
get_headers(IntelliStream_HEADER_FILES)
add_library(IntelliStream SHARED ${IntelliStream_SOURCE_FILES} ${IntelliStream_HEADER_FILES} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(IntelliStream PUBLIC "include")
target_include_directories(IntelliStream PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(IntelliStream PUBLIC ${LIBRARIES})

#Add benchmarks with command
add_subdirectory(benchmark)

#Add tests (based on GTest suits)
add_subdirectory(test)

install(DIRECTORY "include" DESTINATION "/IntelliStream" COMPONENT IntelliStream)

