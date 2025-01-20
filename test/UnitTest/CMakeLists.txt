# Define the executable for testing
add_executable(unit_tests test_compute_engine.cpp)

# Link the executable with the CANDY library
target_link_libraries(unit_tests PRIVATE CANDY Catch2::Catch2WithMain)

# Include directories for headers
target_include_directories(unit_tests PRIVATE ${PROJECT_SOURCE_DIR}/include  ${Catch2_SOURCE_DIR}/src)

# Register the test with Catch2
catch_discover_tests(unit_tests)