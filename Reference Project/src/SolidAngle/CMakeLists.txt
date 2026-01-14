add_library(SolidAngle
    SolidAngle.cpp
)

target_link_libraries(SolidAngle
    PRIVATE
        Utility
        Eigen3::Eigen
)

target_include_directories(SolidAngle
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src
        ${EIGEN3_INCLUDE_DIR}
)