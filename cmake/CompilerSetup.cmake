# cmake/CompilerSetup.cmake

# Check for g++ 11 or higher
find_program(GPP_COMPILER NAMES g++)

if (NOT GPP_COMPILER)
    message(FATAL_ERROR "g++ compiler is required. Please install g++ 11 or higher.")
else ()
    execute_process(COMMAND ${GPP_COMPILER} --version OUTPUT_VARIABLE GPP_VERSION_OUTPUT OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" GPP_VERSION ${GPP_VERSION_OUTPUT})
    string(REGEX MATCH "^([0-9]+)" GPP_VERSION_MAJOR ${GPP_VERSION})

    if (GPP_VERSION_MAJOR LESS 11)
        message(FATAL_ERROR "g++ 11 or higher is required. Please install it and try again.")
    else ()
        message(STATUS "g++ version ${GPP_VERSION} found")
    endif ()
endif ()

# Set Compilation Flags
set(CMAKE_CXX_FLAGS "-Wall -Werror=return-type -Wno-error=unused-variable -Wno-error=unused-parameter")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DNO_RACE_CHECK -DIntelliStream_DEBUG_MODE=1")
set(CMAKE_CXX_FLAGS_RELEASE "-Wno-ignored-qualifiers -Wno-sign-compare -O3")

message(STATUS "Compilation flags set")
