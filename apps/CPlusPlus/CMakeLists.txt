add_sources(Main
        Main.cpp
)

# Get the source files for db_client
get_sources(Main Main_SOURCE_FILES)

# Define the executable for db_client
add_executable(Main ${Main_SOURCE_FILES})

# Link the executable with the CANDY library
target_link_libraries(Main PRIVATE CANDY)

# Include directories for headers
target_include_directories(Main PRIVATE ${PROJECT_SOURCE_DIR}/include)
