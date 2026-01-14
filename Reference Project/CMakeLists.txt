cmake_minimum_required(VERSION 3.10)
project(PhononBTE VERSION 1.0 LANGUAGES CXX)

# =============================================================================
# Configuration and Options
# =============================================================================

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default dimension setting
if(NOT DEFINED DIM)
    set(DIM 2 CACHE STRING "Dimension for the simulation (2 or 3)")
endif()

# Validate DIM value
set(VALID_DIMS "2" "3")
if(NOT DIM IN_LIST VALID_DIMS)
    message(FATAL_ERROR "DIM must be either 2 or 3, got: ${DIM}")
endif()

# Add DIM as compiler definition
add_compile_definitions(DIM=${DIM})

message(STATUS "Building PhononBTE version ${PROJECT_VERSION}")
message(STATUS "Using DIM = ${DIM}")

# =============================================================================
# Compiler Flags and Optimization
# =============================================================================

# Release mode optimization
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -DNDEBUG")

# Debug mode flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -Wall -Wextra")

# Profile mode flags
set(CMAKE_CXX_FLAGS_PROFILE "-pg -O3 -g")
set(CMAKE_EXE_LINKER_FLAGS_PROFILE "-pg")

# =============================================================================
# Dependencies
# =============================================================================

# Find required packages
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(MPI REQUIRED)

# Find METIS library
find_library(METIS_LIBRARY metis PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib)
find_path(METIS_INCLUDE_DIR metis.h PATHS /usr/include /usr/local/include)

if(METIS_LIBRARY AND METIS_INCLUDE_DIR)
    message(STATUS "Found METIS: ${METIS_LIBRARY}")
    message(STATUS "METIS include directory: ${METIS_INCLUDE_DIR}")
else()
    message(WARNING "METIS library not found. Mesh partitioning will not be available.")
    set(METIS_LIBRARY "")
    set(METIS_INCLUDE_DIR "")
endif()

# Verify package versions if needed
if(Eigen3_FOUND)
    message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIRS}")
endif()

if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP: ${OpenMP_CXX_FLAGS}")
endif()

if(yaml-cpp_FOUND)
    message(STATUS "Found yaml-cpp: ${yaml-cpp_VERSION}")
endif()

# =============================================================================
# Output Directories
# =============================================================================

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Add source directory
add_subdirectory(src)


# =============================================================================
# Post-build Configuration
# =============================================================================

message(STATUS "Configuration complete. Build type: ${CMAKE_BUILD_TYPE}")