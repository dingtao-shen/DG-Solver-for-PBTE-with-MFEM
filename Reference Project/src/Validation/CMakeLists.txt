# Create Validation library
add_library(Validation
    MeshPartitionValidator.cpp
)

# Link libraries
target_link_libraries(Validation
    PRIVATE
    Eigen3::Eigen
)

# Include directories
target_include_directories(Validation
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
)

# Add METIS if available
if(METIS_LIBRARY AND METIS_INCLUDE_DIR)
    target_include_directories(Validation PRIVATE ${METIS_INCLUDE_DIR})
    target_link_libraries(Validation PRIVATE ${METIS_LIBRARY})
    target_compile_definitions(Validation PRIVATE HAVE_METIS)
endif()

# Create test executable
add_executable(TestMeshPartition
    TestMeshPartition.cpp
)

target_include_directories(TestMeshPartition
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
)

target_link_libraries(TestMeshPartition
    PRIVATE
    Validation
    SolidAngle
    Utility
    Polynomial
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    yaml-cpp
    stdc++fs
)

# Add METIS if available
if(METIS_LIBRARY AND METIS_INCLUDE_DIR)
    target_include_directories(TestMeshPartition PRIVATE ${METIS_INCLUDE_DIR})
    target_link_libraries(TestMeshPartition PRIVATE ${METIS_LIBRARY})
    target_compile_definitions(TestMeshPartition PRIVATE HAVE_METIS)
endif()

