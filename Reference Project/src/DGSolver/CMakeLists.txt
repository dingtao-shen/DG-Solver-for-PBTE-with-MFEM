# Create DGSolver library
add_library(DGSolver
    PBTE_NonGraySMRT.cpp
    PBTE_NonGraySMRT_MPI.cpp
)

# Link libraries
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

if(OpenMP_CXX_FOUND)
    target_link_libraries(DGSolver
        PRIVATE
        Eigen3::Eigen
        OpenMP::OpenMP_CXX
        Utility
        MPI::MPI_CXX
    )
endif()

# Include directories
target_include_directories(DGSolver
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
    ${MPI_CXX_INCLUDE_DIRS}
)

# Add METIS if available
if(METIS_LIBRARY AND METIS_INCLUDE_DIR)
    target_include_directories(DGSolver PRIVATE ${METIS_INCLUDE_DIR})
    target_link_libraries(DGSolver PRIVATE ${METIS_LIBRARY})
    target_compile_definitions(DGSolver PRIVATE HAVE_METIS)
endif()