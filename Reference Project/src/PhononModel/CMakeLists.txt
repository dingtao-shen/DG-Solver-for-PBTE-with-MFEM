add_library(PhononModel
    NonGraySMRT.cpp
    NonGraySMRT_MPI.cpp
)

target_include_directories(PhononModel
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src
)

target_link_libraries(PhononModel
    PUBLIC
        Utility
        SolidAngle
        OpenMP::OpenMP_CXX
        Eigen3::Eigen
        MPI::MPI_CXX
)

# Add METIS if available
if(METIS_LIBRARY AND METIS_INCLUDE_DIR)
    target_include_directories(PhononModel PRIVATE ${METIS_INCLUDE_DIR})
    target_link_libraries(PhononModel PRIVATE ${METIS_LIBRARY})
    target_compile_definitions(PhononModel PRIVATE HAVE_METIS)
endif()