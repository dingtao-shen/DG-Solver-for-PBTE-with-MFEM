#pragma once

#include <array>
#include <cmath>

namespace dg {

// Minimal gray-equilibrium helpers for quick prototyping.
// These use simple linearized forms easy to replace later.
struct GrayEquilibrium {
    double referenceTemperature{1.0};
    double driftCoefficient{1.0}; // scales <omega, u>

    // Resistive equilibrium (BE, gray, linearized): constant in directions.
    double resistiveBE(double temperature) const {
        // In real models, this would be proportional to energy density at T.
        // Here keep it linear to ease later replacement.
        return temperature / referenceTemperature;
    }

    // Normal equilibrium (drifted BE, gray, linearized): add directional drift.
    double normalDriftedBE(double temperature,
                           const std::array<double,3>& omega,
                           const std::array<double,3>& driftU) const
    {
        const double base = resistiveBE(temperature);
        const double dot = omega[0]*driftU[0] + omega[1]*driftU[1] + omega[2]*driftU[2];
        return base + driftCoefficient * dot;
    }
};

} // namespace dg

