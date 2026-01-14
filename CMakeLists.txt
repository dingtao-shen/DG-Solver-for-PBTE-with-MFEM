cmake_minimum_required(VERSION 3.16)

option(USE_MPI "Use MPI version of MFEM" OFF)

project(DG_Solver_for_PBTE_with_MFEM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(USE_MPI)
    set(CMAKE_CXX_COMPILER mpicxx CACHE STRING "MPI C++ compiler" FORCE)
    set(MFEM_DIR "/usr/local/mfem-mpi/lib/cmake/mfem" CACHE PATH "Path to MFEM config")
    message(STATUS "Building with MPI support")
endif()

find_package(MFEM REQUIRED CONFIG)

message(STATUS "Available MFEM targets:")
get_property(targets GLOBAL PROPERTY IMPORTED_TARGETS)
foreach(target ${targets})
    if(target MATCHES "mfem")
        message(STATUS "  ${target}")
    endif()
endforeach()

if(TARGET MFEM::mfem)
    set(MFEM_TARGET MFEM::mfem)
elseif(TARGET MFEM::mfem-mpi)
    set(MFEM_TARGET MFEM::mfem-mpi)
elseif(TARGET mfem)
    set(MFEM_TARGET mfem)
else()
    message(FATAL_ERROR "No MFEM target found")
endif()

message(STATUS "Using MFEM target: ${MFEM_TARGET}")

find_package(yaml-cpp REQUIRED)

add_library(pbte STATIC
    src/SpatialMesh.cpp
    src/ElementIntegrator.cpp
    src/AngularQuadrature.cpp
    src/AngularSweepOrder.cpp
    src/PhononProperties.cpp
    src/MacroscopicQuantities.cpp
    src/PBTESolver.cpp
    src/Utils.cpp
)

target_include_directories(pbte PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(pbte PUBLIC ${MFEM_TARGET} yaml-cpp)

if(USE_MPI)
    find_package(MPI REQUIRED)
    target_link_libraries(pbte PUBLIC MPI::MPI_CXX)
    target_compile_definitions(pbte PUBLIC USE_MPI)
endif()

add_executable(pbte_demo
    src/PhononBTE.cpp
)

target_link_libraries(pbte_demo PRIVATE pbte ${MFEM_TARGET})

if(USE_MPI)
    target_link_libraries(pbte_demo PRIVATE MPI::MPI_CXX)
endif()