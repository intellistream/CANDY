# cmake/EnableTests.cmake

if (ENABLE_UNIT_TESTS)
    # Add tests that depend on the CANDY library
    add_subdirectory(test)
endif ()
