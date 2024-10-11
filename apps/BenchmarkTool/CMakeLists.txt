add_sources(benchmark_tool
        benchmark_tool.cpp
)

# Get the source files for db_client
get_sources(benchmark_tool DB_CLIENT_SOURCE_FILES)

# Define the executable for db_client
add_executable(benchmark_tool ${DB_CLIENT_SOURCE_FILES})

# Link the executable with the CANDY library
target_link_libraries(benchmark_tool PRIVATE CANDY)

# Include directories for headers
target_include_directories(benchmark_tool PRIVATE ${PROJECT_SOURCE_DIR}/include)
