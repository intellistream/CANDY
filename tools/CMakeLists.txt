cmake_minimum_required(VERSION 3.15)
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/format.cmake)
include_directories(.)
add_executable(run_format run_clang_format.cpp)
