# Add source files for db_client using your add_sources macro
add_sources(db_client
        db_client.cpp
)

# Get the source files for db_client
get_sources(db_client DB_CLIENT_SOURCE_FILES)

# Define the executable for db_client
add_executable(db_client ${DB_CLIENT_SOURCE_FILES})

# Link the executable with the CANDY library
target_link_libraries(db_client PRIVATE CANDY)

# Include directories for headers
target_include_directories(db_client PRIVATE ${PROJECT_SOURCE_DIR}/include)
