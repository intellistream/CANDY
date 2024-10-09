# cmake/EnableLog4cxx.cmake

if (UNIX AND NOT APPLE)
    add_definitions(-DUSELOG4CXX)
    message(STATUS "Log4cxx support enabled (UNIX system detected)")
endif ()

# Find Log4cxx
find_package(Log4cxx REQUIRED)
include_directories(${Log4cxx_INCLUDE_DIR})
set(LIBRARIES ${LIBRARIES} ${Log4cxx_LIBRARY})

message(STATUS "Log4cxx included and linked")
