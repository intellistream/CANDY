# CMakeLists.txt (root)

cmake_minimum_required(VERSION 3.15)
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

# No need to link, as it's a header-only library
target_include_directories(CANDY PUBLIC ${PROJECT_SOURCE_DIR}/ThirdParty/toml11/include)

# Link with external libraries (e.g., Log4cxx)
find_package(Log4cxx REQUIRED)
target_link_libraries(CANDY PUBLIC ${Log4cxx_LIBRARY})

# Set rpath for libCANDY.so to make it available to dependent libraries
set_target_properties(CANDY PROPERTIES INSTALL_RPATH "$ORIGIN")

# Add applications that depend on the CANDY library
add_subdirectory(apps)

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
        COMPONENT CANDY)

# Install headers
install(DIRECTORY "include"
        DESTINATION "${PYTHON_USER_SITE_PACKAGES}/include/CANDY"
        COMPONENT CANDY)

# Add a post-build custom command to automatically run `make install`
add_custom_command(TARGET CANDY
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} --build . --target install
        COMMENT "Installing CANDY to Python user site-packages"
)
