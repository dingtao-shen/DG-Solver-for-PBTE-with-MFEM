#pragma once

#include <vector>
#include <functional>
#include <array>

#include "params.hpp"
#include "quadrature_sn.hpp"

namespace dg {

// Equilibrium providers for Callaway model (gray approximation).
// User can supply simple constants or direction-dependent forms.
struct EquilibriumFields {
    // g_N^eq(omega): drifted BE (normal scattering)
    std::function<double(const std::array<double,3>&)> normal;
    // g_R^eq(omega): BE (resistive scattering)
    std::function<double(const std::array<double,3>&)> resistive;
};

// Callaway's double-relaxation collision operator (gray model).
// Minimal form implements:
//   C[g]_i = - (g_i - g_R,i)/tau_R - (g_i - g_N,i)/tau_N.
class CallawayRTA {
public:
    explicit CallawayRTA(GrayCallawayParams p) : params_(p) {}

    const GrayCallawayParams& params() const { return params_; }

    // Relaxation times given a characteristic length L_char (user-chosen).
    // tau = lambda / v_g, and lambda = Kn * L_char.
    double tauNormal(double L_char) const {
        return (params_.knudsenNormal * L_char) / params_.groupVelocity;
    }
    double tauResistive(double L_char) const {
        return (params_.knudsenResistive * L_char) / params_.groupVelocity;
    }

    // Apply collision operator with separate equilibria.
    void apply(const std::vector<double>& g,
               const EquilibriumFields& eq,
               const SNDirections& directions,
               double L_char,
               std::vector<double>& rhs) const
    {
        const double inv_tau_R = 1.0 / tauResistive(L_char);
        const double inv_tau_N = 1.0 / tauNormal(L_char);
        rhs.resize(g.size());
        for (size_t i = 0; i < g.size(); ++i) {
            const auto& om = directions.omega[i];
            const double gR = eq.resistive ? eq.resistive(om) : 0.0;
            const double gN = eq.normal ? eq.normal(om) : 0.0;
            rhs[i] = - (g[i] - gR) * inv_tau_R
                     - (g[i] - gN) * inv_tau_N;
        }
    }

private:
    GrayCallawayParams params_;
};

} // namespace dg

