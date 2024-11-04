cmake_minimum_required(VERSION 3.15)

# Policy to suppress deprecated warnings
if (POLICY CMP0148)
    cmake_policy(SET CMP0148 NEW)
endif ()

# Download and add pybind11
include(FetchContent)
FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1  # Replace with the desired version
)
FetchContent_MakeAvailable(pybind11)

# Create a shared library for Python
add_library(pycandy MODULE vectordb_pybind.cpp)
set_target_properties(pycandy PROPERTIES PREFIX "")

# Get the Python user site-packages directory using sysconfig
execute_process(
        COMMAND python3 -c "import site; print(site.getusersitepackages())"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
)


# Link the VectorDBCore library to the pycandy module
target_link_libraries(pycandy PRIVATE CANDY)

# Include directories for pybind11 and VectorDB
target_include_directories(pycandy PRIVATE
        ${pybind11_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/include
        ${PYTHON_SITE_PACKAGES}/include
)


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
set_target_properties(pycandy PROPERTIES
        INSTALL_RPATH "$ORIGIN:${TORCH_LIB_PATH}"
)

# Ensure the Python user site-packages directory exists
file(MAKE_DIRECTORY ${PYTHON_SITE_PACKAGES})

# Install the shared library to the user Python site-packages directory
install(TARGETS pycandy
        LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES})

# Add a post-build custom command to always replace the installed .so file
add_custom_command(TARGET pycandy
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:pycandy> ${PYTHON_SITE_PACKAGES}
        COMMENT "Installing pycandy shared library at ${PYTHON_SITE_PACKAGES}"
)

pybind11_add_module(PyCANDYAlgos algos_pybind.cpp)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")
target_link_libraries(PyCANDYAlgos PUBLIC ${LIBRARIES} CANDY ${TORCH_PYTHON_LIBRARY})
set_property(TARGET PyCANDYAlgos  PROPERTY CXX_STANDARD 20)
## Add a post-build custom command to automatically run `make install`
#add_custom_command(TARGET pycandy
#        POST_BUILD
#        COMMAND ${CMAKE_COMMAND} --build . --target install
#        COMMENT "Installing pycandy to Python user site-packages"
#)