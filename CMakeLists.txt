cmake_minimum_required(VERSION 3.16)
project(DG_Solver_for_PBTE_with_MFEM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(MFEM REQUIRED CONFIG)

add_library(pbte STATIC
    src/SpatialMesh.cpp
)

target_include_directories(pbte PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(pbte PUBLIC mfem)

add_executable(pbte_demo
    src/PhononBTE.cpp
)

target_link_libraries(pbte_demo PRIVATE pbte mfem)