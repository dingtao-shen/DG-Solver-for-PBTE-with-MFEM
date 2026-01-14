# Create Polynomial library
add_library(Polynomial
    Polynomial.cpp
)

# Link libraries
target_link_libraries(Polynomial
    PRIVATE
    Eigen3::Eigen
)

# Include directories
target_include_directories(Polynomial
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
) 