macro(get_log_level_value ${CMAKE_PROJECT_NAME}_LOGGING_VALUE)
    if (${${CMAKE_PROJECT_NAME}_LOGGING_LEVEL} STREQUAL "TRACE")
        message("-- Log level is set to TRACE!")
        set(${CMAKE_PROJECT_NAME}_LOGGING_VALUE 6)
    elseif (${${CMAKE_PROJECT_NAME}_LOGGING_LEVEL} STREQUAL "DEBUG")
        message("-- Log level is set to DEBUG!")
        set(${CMAKE_PROJECT_NAME}_LOGGING_VALUE 5)

    elseif (${${CMAKE_PROJECT_NAME}_LOGGING_LEVEL} STREQUAL "INFO")
        message("-- Log level is set to INFO!")
        set(${CMAKE_PROJECT_NAME}_LOGGING_VALUE 4)
    elseif (${${CMAKE_PROJECT_NAME}_LOGGING_LEVEL} STREQUAL "WARN")
        message("-- Log level is set to WARN!")
        set(${CMAKE_PROJECT_NAME}_LOGGING_VALUE 3)

    elseif (${${CMAKE_PROJECT_NAME}_LOGGING_LEVEL} STREQUAL "ERROR")
        message("-- Log level is set to ERROR!")
        set(${CMAKE_PROJECT_NAME}_LOGGING_VALUE 2)

    elseif (${${CMAKE_PROJECT_NAME}_LOGGING_LEVEL} STREQUAL "FATAL_ERROR")
        message("-- Log level is set to FATAL_ERROR!")
        set(${CMAKE_PROJECT_NAME}_LOGGING_VALUE 1)

    else ()
        message(WARNING "-- Could not set ${CMAKE_PROJECT_NAME}_LOGGING_VALUE as ${${CMAKE_PROJECT_NAME}_LOGGING_LEVEL} did not equal any logging level!!!  Defaulting to debug!")
        set(${CMAKE_PROJECT_NAME}_LOGGING_VALUE 5)
    endif ()
endmacro(get_log_level_value ${CMAKE_PROJECT_NAME}_LOGGING_VALUE)

# Macro to add source files to a specific property
macro(add_source PROP_NAME SOURCE_FILES)
    set(SOURCE_FILES_ABSOLUTE)
    foreach (it ${SOURCE_FILES})
        get_filename_component(ABSOLUTE_PATH ${it} ABSOLUTE)
        set(SOURCE_FILES_ABSOLUTE ${SOURCE_FILES_ABSOLUTE} ${ABSOLUTE_PATH})
    endforeach ()

    get_property(OLD_PROP_VAL GLOBAL PROPERTY "${PROP_NAME}_SOURCE_PROP")
    set_property(GLOBAL PROPERTY "${PROP_NAME}_SOURCE_PROP" ${SOURCE_FILES_ABSOLUTE} ${OLD_PROP_VAL})
endmacro()

# Macro for adding multiple sources
macro(add_sources TARGET_NAME)
    add_source(${TARGET_NAME} "${ARGN}")
endmacro()

# Macro to get source files for a specific target
macro(get_sources TARGET_NAME SOURCE_FILES)
    get_property(SOURCE_FILES_LOCAL GLOBAL PROPERTY "${TARGET_NAME}_SOURCE_PROP")
    set(${SOURCE_FILES} ${SOURCE_FILES_LOCAL})
endmacro()

macro(get_headers HEADER_FILES)
    file(GLOB_RECURSE ${HEADER_FILES} "include/*.h" "include/*.hpp" "include/*.cuh")
endmacro()