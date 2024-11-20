add_sources(OnlineInsert
        Main.cpp
)

add_source(MultiRW
        MultiRW.cpp
)

# add candy_path definition
add_definitions(-DCANDY_PATH="${PROJECT_SOURCE_DIR}")

# Get the source files for db_client
get_sources(OnlineInsert OnlineInsert_SOURCE_FILES)
get_sources(MultiRW MultiRW_SOURCE_FILES)

# Define the executable for db_client
add_executable(OnlineInsert ${OnlineInsert_SOURCE_FILES})
add_executable(MultiRW ${MultiRW_SOURCE_FILES})

# Link the executable with the CANDY library
target_link_libraries(OnlineInsert PRIVATE CANDY)
target_link_libraries(MultiRW PRIVATE CANDY)

# Include directories for headers
target_include_directories(OnlineInsert PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(MultiRW PRIVATE ${PROJECT_SOURCE_DIR}/include)
