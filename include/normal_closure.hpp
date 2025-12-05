#pragma once

#include <array>
#include <vector>
#include "quadrature_sn.hpp"

namespace dg {

// Compute drift vector u such that the first angular moment
// sum_i w_i omega_i (g_i - gN_eq(T, omega_i, u)) = 0.
// Using GrayEquilibrium linearization: gN_eq = base(T) + alpha * (omega · u).
// Then: sum w omega (g - base) = alpha * sum w omega (omega · u)
// => b = alpha * M u, with M = sum w (omega ⊗ omega).
// Returns u = (1/alpha) M^{-1} b.
inline std::array<double,3> computeNormalDriftLinearized(
    const SNDirections& dirs,
    const std::vector<double>& g,
    double base_value,
    double alpha)
{
    std::array<double,3> u{0.0, 0.0, 0.0};
    const int m = static_cast<int>(dirs.omega.size());
    if (m == 0 || alpha == 0.0) { return u; }

    // Assemble 3x3 M and 3x1 b for up to 3D; truncate by dim.
    double M[3][3] = {{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
    double b[3] = {0.0, 0.0, 0.0};
    const int dim = dirs.dimension;
    for (int i = 0; i < m; ++i) {
        const auto &om = dirs.omega[i];
        const double wi = dirs.weight[i];
        const double diff = g[i] - base_value;
        for (int a = 0; a < dim; ++a) {
            b[a] += wi * om[a] * diff;
            for (int c = 0; c < dim; ++c) {
                M[a][c] += wi * om[a] * om[c];
            }
        }
    }
    // Solve M x = b / alpha in dim x dim (use naive 3x3 Gauss).
    // Build augmented matrix A = [M | rhs] and eliminate.
    double A[3][4] = {
        {M[0][0], M[0][1], M[0][2], b[0]/alpha},
        {M[1][0], M[1][1], M[1][2], b[1]/alpha},
        {M[2][0], M[2][1], M[2][2], b[2]/alpha}
    };
    // Gaussian elimination for 'dim'.
    for (int k = 0; k < dim; ++k) {
        // Pivot (no partial pivoting for minimal demo).
        double piv = A[k][k];
        if (std::abs(piv) < 1e-14) continue;
        for (int j = k; j <= dim; ++j) A[k][j] /= piv;
        for (int i = 0; i < dim; ++i) {
            if (i == k) continue;
            double f = A[i][k];
            for (int j = k; j <= dim; ++j) {
                A[i][j] -= f * A[k][j];
            }
        }
    }
    for (int a = 0; a < dim; ++a) { u[a] = A[a][dim]; }
    for (int a = dim; a < 3; ++a) { u[a] = 0.0; }
    return u;
}

} // namespace dg

