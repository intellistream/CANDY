# cmake/AddTests.cmake

macro(add_all_tests TEST_DIR)
    # Recursively search for all *_test.cpp files in the TEST_DIR and its subdirectories
    file(GLOB_RECURSE TEST_SOURCES "${TEST_DIR}/*_test.cpp")

    foreach (TEST_SRC ${TEST_SOURCES})
        # Extract the test name from the filename (remove the directory and extension)
        get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)

        # Create an executable for the test
        add_executable(${TEST_NAME} ${TEST_SRC})

        # Link the CANDY library to the test executable
        target_link_libraries(${TEST_NAME} PRIVATE CANDY)

        # Register the test with CTest
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endforeach ()
endmacro()
