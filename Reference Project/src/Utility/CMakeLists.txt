# Create Utility library
add_library(Utility
    math_utils.cpp
)

# Link libraries
target_link_libraries(Utility
    PRIVATE
    Eigen3::Eigen
)

# Include directories
target_include_directories(Utility
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
) 