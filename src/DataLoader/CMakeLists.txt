add_sources(CANDY
        AbstractDataLoader.cpp
        DataLoaderTable.cpp
        RandomDataLoader.cpp
        FVECSDataLoader.cpp
)
#if (ENABLE_HDF5)
#    add_sources(HDF5DataLoader.cpp)
#endif ()