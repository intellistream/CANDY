add_sources(stream_examples
        main.cpp
)

# Get the source files for db_client
get_sources(stream_examples DB_CLIENT_SOURCE_FILES)

# Define the executable for db_client
add_executable(stream_examples ${DB_CLIENT_SOURCE_FILES})

# Link the executable with the CANDY library
target_link_libraries(stream_examples PRIVATE CANDY)

# Include directories for headers
target_include_directories(stream_examples PRIVATE ${PROJECT_SOURCE_DIR}/include)
