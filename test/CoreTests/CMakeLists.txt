# Define the executable for testing
add_executable(core_tests core_tests.cpp)

# Link the executable with the CANDY library
target_link_libraries(core_tests PRIVATE CANDY Catch2::Catch2WithMain)

# Include directories for headers
target_include_directories(core_tests PRIVATE ${PROJECT_SOURCE_DIR}/include  ${Catch2_SOURCE_DIR}/src)

# Register the test with Catch2
catch_discover_tests(core_tests)