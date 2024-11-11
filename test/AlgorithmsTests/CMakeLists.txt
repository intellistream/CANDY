# add candy_path definition
add_definitions(-DCANDY_PATH="${PROJECT_SOURCE_DIR}")
add_catch_test(KdTreeTests KdTreeTests.cpp)
add_catch_test(HNSWTests HNSWTests.cpp)
add_catch_test(LSHTests LSHTests.cpp)
