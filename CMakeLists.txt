cmake_minimum_required(VERSION 3.16)

project(DG-Solver-for-PBTE-with-MFEM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

file(GLOB_RECURSE PROJECT_SOURCES
    ${PROJECT_SOURCE_DIR}/src/*.c
    ${PROJECT_SOURCE_DIR}/src/*.cc
    ${PROJECT_SOURCE_DIR}/src/*.cpp
    ${PROJECT_SOURCE_DIR}/src/*.cxx
)

if(EXISTS ${PROJECT_INCLUDE_DIR})
    include_directories(${PROJECT_INCLUDE_DIR})
endif()

find_package(MPI QUIET)
if(MPI_CXX_FOUND)
    message(STATUS "MPI found: ${MPI_CXX_COMPILER}")
endif()
 
find_package(MFEM QUIET)
if(MFEM_FOUND)
    message(STATUS "MFEM found: ${MFEM_DIR}")
endif()

if(PROJECT_SOURCES)
    add_executable(DG4PBTE ${PROJECT_SOURCES})

    if(MPI_CXX_FOUND)
        target_link_libraries(DG4PBTE PRIVATE MPI::MPI_CXX)
    endif()

    if(MFEM_FOUND)
        target_link_libraries(DG4PBTE PRIVATE mfem)
        target_compile_definitions(DG4PBTE PRIVATE MFEM_USE_CONFIG_H)
    endif()

    if(EXISTS ${PROJECT_INCLUDE_DIR})
        target_include_directories(DG4PBTE PRIVATE ${PROJECT_INCLUDE_DIR})
    endif()
else()
    message(WARNING "No sources found under src/. Configuration will succeed, but compile_commands.json may be empty until sources are added.")
endif()


