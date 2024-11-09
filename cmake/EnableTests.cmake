# cmake/EnableTests.cmake

option(ENABLE_UNIT_TESTS "Enable unit tests" ON)
message(STATUS "Enable testing: ${ENABLE_UNIT_TESTS}")
if (${ENABLE_UNIT_TESTS})
    include(FetchContent)

    FetchContent_Declare(
            Catch2
            GIT_REPOSITORY https://gitcode.com/gh_mirrors/ca/Catch2.git
            GIT_TAG v3.4.0
    )
    FetchContent_MakeAvailable(Catch2)

    list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
    include(CTest)
    include(Catch)

    enable_testing()  # Enables CTest functionality
endif ()