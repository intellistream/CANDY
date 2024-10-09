# ck_wg +4
# Copyright 2010 by Kitware, Inc.
# All Rights Reserved. Please refer to KITWARE_LICENSE.TXT for licensing information,
# or contact General Counsel, Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

# This script locates the system-installed Log4cxx library.
# The following variables will be set:
# Log4cxx_FOUND       - Set to true if Log4cxx is found
# Log4cxx_INCLUDE_DIR - The path to the Log4cxx header files
# Log4cxx_LIBRARY     - The full path to the Log4cxx library

if (Log4cxx_DIR)
    find_package(Log4cxx NO_MODULE)
elseif (NOT Log4cxx_FOUND)
    message(STATUS "Searching for log4cxx/logger.h header file...")
    find_path(Log4cxx_INCLUDE_DIR log4cxx/logger.h)

    message(STATUS "Searching for Log4cxx library...")
    find_library(Log4cxx_LIBRARY NAMES log4cxx)

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(Log4cxx REQUIRED_VARS Log4cxx_INCLUDE_DIR Log4cxx_LIBRARY)

    if (Log4cxx_FOUND)
        set(Log4cxx_FOUND TRUE)
        message(STATUS "Log4cxx found and configured.")
    else()
        message(FATAL_ERROR "Log4cxx is required but not found. Please install it (e.g., sudo apt-get install -y liblog4cxx-dev) and try again.")
    endif()
endif()
