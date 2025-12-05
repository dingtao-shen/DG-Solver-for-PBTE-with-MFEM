#pragma once

#include <array>
#include <memory>
#include <mfem.hpp>
#include "flux_upwind.hpp"

namespace dg {

// Constant velocity coefficient v = v_g * omega (dimension 2 or 3).
class VelocityCoefficient : public mfem::VectorCoefficient {
public:
    VelocityCoefficient(int dim, double vg, const std::array<double,3>& omega)
        : mfem::VectorCoefficient(dim), dim_(dim)
    {
        v_[0] = vg * omega[0];
        v_[1] = (dim_ > 1) ? vg * omega[1] : 0.0;
        v_[2] = (dim_ > 2) ? vg * omega[2] : 0.0;
    }

    void Eval(mfem::Vector &V, mfem::ElementTransformation &, const mfem::IntegrationPoint &) override
    {
        V.SetSize(dim_);
        for (int i = 0; i < dim_; ++i) { V[i] = v_[i]; }
    }

private:
    int dim_;
    double v_[3] {0.0, 0.0, 0.0};
};

// Build a DG advection bilinear form for a single discrete direction.
// Currently includes the volumetric term (w · grad u, v). Upwind face flux
// integrators will be added in subsequent steps.
inline std::unique_ptr<mfem::BilinearForm>
buildDGAdvectionForm(mfem::FiniteElementSpace& fes,
                     const std::array<double,3>& omega,
                     double group_velocity,
                     const void* /*unused_bdr*/ = nullptr)
{
    auto a = std::make_unique<mfem::BilinearForm>(&fes);
    // Note: Convection term will be added by the caller with a
    // heap-allocated VectorCoefficient to ensure lifetime safety.

    // Add interior upwind numerical flux (boundary handled separately).
    a->AddInteriorFaceIntegrator(
        new UpwindFaceIntegrator(fes.GetMesh()->Dimension(), group_velocity, omega, nullptr));
    // Add boundary upwind numerical flux (outflow matrix term).
    a->AddBdrFaceIntegrator(
        new UpwindFaceIntegrator(fes.GetMesh()->Dimension(), group_velocity, omega, nullptr));

    return a;
}

} // namespace dg

