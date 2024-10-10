# cmake/EnableTests.cmake

option(ENABLE_UNIT_TESTS "Enable unit tests" ON)
message(STATUS "Enable testing: ${ENABLE_UNIT_TESTS}")
if(${ENABLE_UNIT_TESTS})
    enable_testing()  # Enables CTest functionality
endif ()