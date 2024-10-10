# CMakeLists.txt (root)

cmake_minimum_required(VERSION 3.14)
project(CANDY CXX)

# Add the cmake/ directory to CMAKE_MODULE_PATH
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Include necessary modules and options
include(cmake/macros.cmake)
include(cmake/DefaultOptions.cmake)

# Set C++ Standard
include(cmake/SetCXXStandard.cmake)

# Determine OS and enable Log4cxx
include(cmake/EnableLog4cxx.cmake)

# Compiler and build flag settings
include(cmake/CompilerSetup.cmake)

# Logging setup (if needed)
include(cmake/LoggingSetup.cmake)

# Collect Source Files
add_subdirectory(src)

# Assume that all the source files under src/ will be included in the library
get_sources(CANDY CANDY_SOURCE_FILES)
message(STATUS "CANDY_SOURCE_FILES: ${CANDY_SOURCE_FILES}")

get_headers(CANDY_HEADER_FILES)

# Create the CANDY library from all the source files
add_library(CANDY SHARED ${CANDY_SOURCE_FILES} ${CANDY_HEADER_FILES})

# Add the include directory with absolute path
target_include_directories(CANDY PUBLIC ${PROJECT_SOURCE_DIR}/include) # Include the "include/" directory as a system path
target_include_directories(CANDY PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")

# Link with external libraries (e.g., Log4cxx)
target_link_libraries(CANDY PUBLIC ${Log4cxx_LIBRARY})

# Add applications that depend on the CANDY library
add_subdirectory(apps)

# Enable Unit Tests
include(cmake/EnableTests.cmake)

# Install headers
install(DIRECTORY "include" DESTINATION "/CANDY" COMPONENT CANDY)
