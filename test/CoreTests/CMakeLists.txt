# Define the executable for testing
add_executable(core_tests core_tests.cpp)

# Link the executable with the CANDY library
target_link_libraries(core_tests PRIVATE CANDY)

# Include directories for headers
target_include_directories(core_tests PRIVATE ${PROJECT_SOURCE_DIR}/include)

# Register the test with CTest
add_test(NAME core_tests COMMAND core_tests)