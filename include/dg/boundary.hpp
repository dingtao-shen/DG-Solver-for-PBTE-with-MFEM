#pragma once

#include <array>
#include <functional>
#include <unordered_map>
#include <mfem.hpp>

namespace dg {

enum class BoundaryType {
    Isothermal,   // Dirichlet-type for distribution corresponding to BE(Tw)
    Adiabatic,    // Zero normal energy flux
    Specular,     // Specular reflection
    Diffuse       // Diffuse reflection to wall BE(Tw)
};

struct BoundaryData {
    BoundaryType type{BoundaryType::Adiabatic};
    // Wall temperature (for Isothermal/Diffuse).
    double wallTemperature{1.0};
    // Wall drift velocity for normal-drift equilibrium if needed (rare).
    std::array<double,3> wallDrift{0.0, 0.0, 0.0};
    // For mixed models (not used now).
    double emissivity{1.0};
};

// Map MFEM boundary attribute -> boundary data.
using BoundaryConditionMap = std::unordered_map<int, BoundaryData>;

// Helper to detect inflow for a given discrete direction at a boundary face:
// inflow if (v_g * omega) · n < 0.
inline bool isInflow(const std::array<double,3>& omega,
                     double vg,
                     const mfem::Vector& normal)
{
    const double an = vg*(omega[0]*normal[0] +
                          (normal.Size() > 1 ? omega[1]*normal[1] : 0.0) +
                          (normal.Size() > 2 ? omega[2]*normal[2] : 0.0));
    return an < 0.0;
}

} // namespace dg

