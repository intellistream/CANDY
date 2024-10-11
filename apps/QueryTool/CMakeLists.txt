add_sources(query_tool
        query_tool.cpp
)

# Get the source files for query_tool
get_sources(query_tool DB_CLIENT_SOURCE_FILES)

# Define the executable for query_tool
add_executable(query_tool ${DB_CLIENT_SOURCE_FILES})

# Link the executable with the CANDY library
target_link_libraries(query_tool PRIVATE CANDY)

# Include directories for headers
target_include_directories(query_tool PRIVATE ${PROJECT_SOURCE_DIR}/include)
