# adding the Google_Tests_run target
add_executable(Google_Tests_run
        SystemTest/SimpleTest.cpp
        )

# linking Google_Tests_run with IntelliStream_lib which will be tested
target_link_libraries(Google_Tests_run gtest_main IntelliStream)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/datasets
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

gtest_discover_tests(Google_Tests_run)