find_program(CLANG_FORMAT_BIN 
    NAMES clang-format clang-format-18 clang-format-14 clang-format-11
    DOC "Path to clang-format executable"
)

if (CLANG_FORMAT_BIN)
    message(STATUS "clang-format found: ${CLANG_FORMAT_BIN}")
else()
    message(STATUS "clang-format not found or clang-format version is not 11, 14 or 18")
    set(CLANG_FORMAT_BIN "not_found")
endif()

add_custom_target(
    format 
    COMMAND run_format ${CLANG_FORMAT_BIN} --source_dirs apps test src include python_bindings
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
add_dependencies(format run_format)

