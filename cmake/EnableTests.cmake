# cmake/EnableTests.cmake

option(ENABLE_UNIT_TESTS "Enable unit tests" ON)
message(STATUS "Enable testing: ${ENABLE_UNIT_TESTS}")

if (ENABLE_UNIT_TESTS)
    # Add tests that depend on the CANDY library
    add_subdirectory(test)
endif()
