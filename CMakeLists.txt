cmake_minimum_required(VERSION 3.15)
project(CANDY CXX)
# Logging setup (if needed)
#include(cmake/LoggingSetup.cmake)
include(cmake/FindCuda.cmake)
include(cmake/FindTorch.cmake)
find_package(Torch REQUIRED)
# Add the cmake/ directory to CMAKE_MODULE_PATH
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

# Include necessary modules and options
include(cmake/macros.cmake)
include(cmake/DefaultOptions.cmake)
include(cmake/CompilerSetup.cmake)
add_subdirectory(tools)
# # Find python development files
find_package(Python REQUIRED COMPONENTS Interpreter Development)
include_directories(${Python3_INCLUDE_DIRS})

# Link with external libraries (e.g., Log4cxx)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -Wunused-function")
set(LIBRARIES ${LIBRARIES} ${TORCH_LIBRARIES})
# Collect Source Files
add_subdirectory(src)

# Assume that all the source files under src/ will be included in the library
get_sources(CANDY CANDY_SOURCE_FILES)
message(STATUS "CANDY_SOURCE_FILES: ${CANDY_SOURCE_FILES}")

get_headers(CANDY_HEADER_FILES)
# Create the CANDY library from all the source files

add_library(CANDY SHARED ${CANDY_SOURCE_FILES} ${CANDY_HEADER_FILES}
        include/DataLoader/DataLoaderTable.hpp
)

# Add the include directory with absolute path
target_include_directories(CANDY PUBLIC ${PROJECT_SOURCE_DIR}/include) # Include the "include/" directory as a system path
target_include_directories(CANDY PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")

message(STATUS "Linked libs: ${LIBRARIES}")
target_link_libraries(CANDY PUBLIC ${TORCH_LIBRARIES})

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
if (ENABLE_AUTO_GLOBAL_PYTHON_PKG)
    # Add a post-build custom command to automatically run `make install`
    add_custom_command(TARGET CANDY
            POST_BUILD
            COMMAND cd ${PROJECT_SOURCE_DIR} && python3 ${PROJECT_SOURCE_DIR}/installation/python_package_setup/setup.py install --user
            COMMENT "Installing compiled  python packages"
    )
endif ()

# check if the user is trying to build in the source directory
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" PATH_TO_CMAKELISTS_TXT)

if (EXISTS "${PATH_TO_CMAKELISTS_TXT}")
    message(FATAL_ERROR "Run CMake from a build subdirectory! \"mkdir build ; cd build ; cmake ..\" \
    Some junk files were created in this folder (CMakeCache.txt, CMakeFiles); you should delete those.")
endif ()
