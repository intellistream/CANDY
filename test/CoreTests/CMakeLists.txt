# Add the source files for core_tests using the custom macros
add_sources(core_tests
        core_tests.cpp
)

# Get the source files added to the core_tests property
get_sources(core_tests CORE_TEST_SOURCE_FILES)

# Define the executable for testing
add_executable(core_tests ${CORE_TEST_SOURCE_FILES})

# Link the executable with the CANDY library
target_link_libraries(core_tests PRIVATE CANDY)

# Include directories for headers
target_include_directories(core_tests PRIVATE ${PROJECT_SOURCE_DIR}/include)
