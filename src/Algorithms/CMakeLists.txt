add_subdirectory(HNSW)
add_subdirectory(KDTree)
add_subdirectory(KNN)
add_subdirectory(LSH)
add_subdirectory(Utils)
add_subdirectory(FlatGPUIndex)

add_sources(CANDY
        ANNSBase.cpp
)

if (ENABLE_CUDA)
    add_subdirectory(SONG)
endif ()