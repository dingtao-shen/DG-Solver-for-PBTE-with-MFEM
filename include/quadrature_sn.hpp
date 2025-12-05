#pragma once

#include <array>
#include <vector>

namespace dg {

// Stores discrete ordinates directions and weights (simple placeholder sets).
struct SNDirections {
    int dimension{3};
    // Store 3 components even in 2D; ignore the extra component for 2D.
    std::vector<std::array<double, 3>> omega;  // unit directions
    std::vector<double> weight;                // positive weights summing to 1
};

// Very small direction sets for a minimal demo (not production SN sets).
inline SNDirections makeLevelSymmetricSN(int order, int dim) {
    SNDirections dir;
    dir.dimension = dim;

    if (dim == 2) {
        // Minimal 2D set: 4 cardinal directions with equal weights.
        dir.omega = {
            { { 1.0,  0.0, 0.0 } },
            { { -1.0, 0.0, 0.0 } },
            { { 0.0,  1.0, 0.0 } },
            { { 0.0, -1.0, 0.0 } },
        };
        dir.weight = {0.25, 0.25, 0.25, 0.25};
        return dir;
    }

    // Default to 3D: 6 axis-aligned directions with equal weights.
    dir.omega = {
        { { 1.0,  0.0,  0.0 } },
        { { -1.0, 0.0,  0.0 } },
        { { 0.0,  1.0,  0.0 } },
        { { 0.0, -1.0,  0.0 } },
        { { 0.0,  0.0,  1.0 } },
        { { 0.0,  0.0, -1.0 } },
    };
    dir.weight = {1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0};
    return dir;
}

}  // namespace dg

