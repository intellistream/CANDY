cmake_minimum_required(VERSION 3.15)

# Policy to suppress deprecated warnings
if(POLICY CMP0148)
    cmake_policy(SET CMP0148 NEW)
endif()

# Download and add pybind11
include(FetchContent)
FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1  # Replace with the desired version
)
FetchContent_MakeAvailable(pybind11)

# Create a shared library for Python
add_library(pyvectordb MODULE vectordb_pybind.cpp)
set_target_properties(pyvectordb PROPERTIES PREFIX "")

# Get the Python user site-packages directory using sysconfig
execute_process(
        COMMAND python3 -c "import site; print(site.getusersitepackages())"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Link the VectorDBCore library to the pyvectordb module
target_link_libraries(pyvectordb PRIVATE CANDY)

# Include directories for pybind11 and VectorDB
target_include_directories(pyvectordb PRIVATE
        ${pybind11_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/include
        ${PYTHON_SITE_PACKAGES}/include
)

# Set rpath to include the directory of libCANDY.so
set_target_properties(pyvectordb PROPERTIES INSTALL_RPATH "$ORIGIN")

# Ensure the Python user site-packages directory exists
file(MAKE_DIRECTORY ${PYTHON_SITE_PACKAGES})

# Install the shared library to the user Python site-packages directory
install(TARGETS pyvectordb
        LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES})

## Add a post-build custom command to automatically run `make install`
#add_custom_command(TARGET pyvectordb
#        POST_BUILD
#        COMMAND ${CMAKE_COMMAND} --build . --target install
#        COMMENT "Installing pyvectordb to Python user site-packages"
#)