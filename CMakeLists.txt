cmake_minimum_required(VERSION 3.18)
project(CANDY CXX)
# Logging setup (if needed)
#include(cmake/LoggingSetup.cmake)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()
option(ENABLE_CUDA "Enable cuda" ON)

include(cmake/FindCuda.cmake)
include(cmake/FindTorch.cmake)
find_package(Torch REQUIRED)
# Add the cmake/ directory to CMAKE_MODULE_PATH
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Include necessary modules and options
include(cmake/macros.cmake)
include(cmake/DefaultOptions.cmake)
include(cmake/CompilerSetup.cmake)
add_subdirectory(tools)
# # Find python development files
find_package(Python REQUIRED COMPONENTS Interpreter Development)
include_directories(${Python3_INCLUDE_DIRS})

# Option to enable/disable CUDA
message(STATUS "CUDA enabled: ${ENABLE_CUDA}")
if (ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_ARCHITECTURES OFF)
    add_definitions(-DENABLE_CUDA=1)
endif ()

# Link with external libraries (e.g., Log4cxx)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -Wunused-function")
set(LIBRARIES ${LIBRARIES} ${TORCH_LIBRARIES})
# Collect Source Files
add_subdirectory(src)

include(${PROJECT_SOURCE_DIR}/cmake/third-party.cmake)
# Assume that all the source files under src/ will be included in the library
get_sources(CANDY CANDY_SOURCE_FILES)
message(STATUS "CANDY_SOURCE_FILES: ${CANDY_SOURCE_FILES}")

get_headers(CANDY_HEADER_FILES)

# Set the CUDA library path for linker flags
set(CUDA_LIB_PATH "/usr/local/cuda/lib64")

# Create the CANDY library from all the source files

if (ENABLE_CUDA)
    set_source_files_properties(src/Algorithms/AlgorithmTable.cpp
            PROPERTIES
            LANGUAGE CUDA
    )
endif ()

add_library(CANDY SHARED ${CANDY_SOURCE_FILES} ${CANDY_HEADER_FILES}
        include/DataLoader/DataLoaderTable.hpp
        src/Algorithms/AlgorithmTable.cpp
)

# Add the include directory with absolute path
target_include_directories(CANDY PUBLIC ${PROJECT_SOURCE_DIR}/include) # Include the "include/" directory as a system path
target_include_directories(CANDY PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")


# Add the CUDA library path for all targets
link_directories(${CUDA_LIB_PATH})

# Ensure library paths are included for shared library builds
target_link_directories(CANDY PUBLIC ${CUDA_LIB_PATH})

# Add linker flags for executables and shared libraries
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${CUDA_LIB_PATH}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L${CUDA_LIB_PATH}")

if (ENABLE_CUDA)
    set(LIBRARIES ${LIBRARIES} cublas cudart)
    set_target_properties(CANDY PROPERTIES
            CUDA_STANDARD 20
            CXX_STANDARD 20
    )
else ()
    set_target_properties(CANDY PROPERTIES
            CXX_STANDARD 20
    )
endif ()

message(STATUS "Linked libs: ${LIBRARIES}")
target_link_libraries(CANDY PUBLIC ${LIBRARIES})

# Set rpath for libCANDY.so to make it available to dependent libraries
# Use Python to get the torch library path dynamically
execute_process(
        COMMAND python3 -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
        OUTPUT_VARIABLE TORCH_LIB_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Print the torch library path for debugging
message(STATUS "PyTorch library path: ${TORCH_LIB_PATH}")

# Set RPATH to use $ORIGIN and the dynamic PyTorch library path
set_target_properties(CANDY PROPERTIES
        INSTALL_RPATH "$ORIGIN:${TORCH_LIB_PATH}"
)

# Add applications that depend on the CANDY library
add_subdirectory(apps)

# Copy the config directory to the build output directory
file(COPY ${CMAKE_SOURCE_DIR}/config DESTINATION ${CMAKE_BINARY_DIR})

# Option to enable/disable building tests
option(ENABLE_TESTS "Enable unit tests" ON)

if (ENABLE_TESTS)
    include(cmake/EnableTests.cmake)
    add_subdirectory(test)
endif ()

# Python bindings
add_subdirectory(python_bindings/pybind11_bindings)

# Install the core shared library (libCANDY.so) to user site-packages
execute_process(
        COMMAND python3 -c "import site; print(site.getusersitepackages())"
        OUTPUT_VARIABLE PYTHON_USER_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Install targets
install(TARGETS CANDY
        LIBRARY DESTINATION ${PYTHON_USER_SITE_PACKAGES}
        COMPONENT CANDY
)

# Install headers
install(DIRECTORY "include"
        DESTINATION "${PYTHON_USER_SITE_PACKAGES}/include/CANDY"
        COMPONENT CANDY
)

option(ENABLE_AUTO_GLOBAL_PYTHON_PKG
        "Enable automatically add and update global packages to python"
        ON
)

# check if the user is trying to build in the source directory
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" PATH_TO_CMAKELISTS_TXT)

if (EXISTS "${PATH_TO_CMAKELISTS_TXT}")
    message(FATAL_ERROR "Run CMake from a build subdirectory! \"mkdir build ; cd build ; cmake ..\" \
    Some junk files were created in this folder (CMakeCache.txt, CMakeFiles); you should delete those.")
endif ()
