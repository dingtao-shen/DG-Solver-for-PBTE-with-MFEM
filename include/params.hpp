#pragma once

#include <cstddef>

namespace dg {

// Parameters for gray (frequency-averaged) Callaway double-relaxation model.
struct GrayCallawayParams {
    // Spatial dimension (2 or 3).
    int dimension{3};

    // Group velocity v_g (assumed isotropic in gray model).
    double groupVelocity{1.0};

    // Non-dimensional Knudsen numbers based on a chosen characteristic length L_char:
    // Kn_N = lambda_N / L_char, Kn_R = lambda_R / L_char.
    double knudsenNormal{1.0};
    double knudsenResistive{1.0};

    // Effective volumetric heat capacity (can be used for source/diagnostics).
    double heatCapacity{1.0};
};

}  // namespace dg

