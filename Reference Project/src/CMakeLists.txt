# Add subdirectories for each module
add_subdirectory(Utility)
add_subdirectory(Polynomial)
add_subdirectory(SolidAngle)
add_subdirectory(PhononModel)
add_subdirectory(DGSolver)
add_subdirectory(Validation)

add_executable(PhononBTE PhononBTE/PhononBTE.cpp)
target_include_directories(PhononBTE
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(PhononBTE
    PRIVATE
    Utility
    Polynomial
    SolidAngle
    DGSolver
    PhononModel
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    yaml-cpp
    stdc++fs
)

# Add METIS if available
if(METIS_LIBRARY AND METIS_INCLUDE_DIR)
    target_include_directories(PhononBTE PRIVATE ${METIS_INCLUDE_DIR})
    target_link_libraries(PhononBTE PRIVATE ${METIS_LIBRARY})
    target_compile_definitions(PhononBTE PRIVATE HAVE_METIS)
endif()