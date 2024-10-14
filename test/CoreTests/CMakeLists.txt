# Define the executable for testing
macro(add_catch_test appName SOURCE_FILES)
    add_executable(${appName} ${SOURCE_FILES})
    # Link the executable with the CANDY library
    target_link_libraries(${appName} PRIVATE CANDY Catch2::Catch2WithMain)
    target_include_directories(${appName} PRIVATE ${PROJECT_SOURCE_DIR}/include ${Catch2_SOURCE_DIR}/src)
    catch_discover_tests(${appName})
endmacro()
add_catch_test(core_tests core_tests.cpp)
add_catch_test(torch_tests torch_tests.cpp)
