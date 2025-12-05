#pragma once

#include <memory>
#include <mfem.hpp>
#include "quadrature_sn.hpp"
#include "params.hpp"
#include "dg/transport.hpp"
#include "callaway_rta.hpp"

namespace dg {

// Assemble a linear steady system for a single direction i:
// (w·grad) g_i + sigma * g_i = sigma_R * g_R^eq + sigma_N * g_N^eq
// where sigma_{R,N} = 1/tau_{R,N}(L_char). Discretization:
// A_i = Convection(vol) + sigma * Mass;  b_i = Mass * (sigma_R g_R^eq + sigma_N g_N^eq).
struct DirectionSystem {
    std::unique_ptr<mfem::BilinearForm> Aform;
    std::unique_ptr<mfem::LinearForm> bform;
    std::unique_ptr<mfem::GridFunction> solution;
    // Keep owned coefficients alive for the lifetime of the system.
    std::vector<std::unique_ptr<mfem::Coefficient>> owned_coeffs;
    std::vector<std::unique_ptr<mfem::VectorCoefficient>> owned_vcoeffs;
    // Keep owned boundary maps alive (for boundary face integrators).
    std::vector<std::unique_ptr<dg::BoundaryConditionMap>> owned_bmaps;
};

inline DirectionSystem buildDirectionSystem(mfem::FiniteElementSpace& fes,
                                            const SNDirections& dirs,
                                            int idir,
                                            const GrayCallawayParams& params,
                                            double L_char,
                                            const EquilibriumFields& eq, // kept for inflow default
                                            const GrayEquilibrium& geq,  // used for spatial T
                                            const mfem::GridFunction* T_field, // temperature field
                                            const std::array<double,3>& drift_u) // constant drift
{
    DirectionSystem sys;
    // Convection + sigma Mass
    sys.Aform = buildDGAdvectionForm(fes, dirs.omega[idir], params.groupVelocity, nullptr);
    // Add convection with owned VectorCoefficient to ensure lifetime safety
    auto vel = std::make_unique<dg::VelocityCoefficient>(fes.GetMesh()->Dimension(),
                                                         params.groupVelocity,
                                                         dirs.omega[idir]);
    sys.Aform->AddDomainIntegrator(new mfem::ConvectionIntegrator(*vel));
    sys.owned_vcoeffs.push_back(std::move(vel));
    const double sigmaR = params.groupVelocity > 0.0 ? (1.0 / ((params.knudsenResistive * L_char) / params.groupVelocity)) : 0.0;
    const double sigmaN = params.groupVelocity > 0.0 ? (1.0 / ((params.knudsenNormal    * L_char) / params.groupVelocity)) : 0.0;
    const double sigma = sigmaR + sigmaN;
    auto cc_sigma = std::make_unique<mfem::ConstantCoefficient>(sigma);
    sys.Aform->AddDomainIntegrator(new mfem::MassIntegrator(*cc_sigma));
    sys.owned_coeffs.push_back(std::move(cc_sigma));

    // RHS: Mass * (sigma_R g_R^eq + sigma_N g_N^eq)
    class EqCoefficient : public mfem::Coefficient {
    public:
        EqCoefficient(const SNDirections& dirs, int idir,
                      double sR, double sN,
                      const GrayEquilibrium& geq,
                      const mfem::GridFunction* Tfield,
                      const std::array<double,3>& drift_u)
            : dirs_(dirs), idir_(idir), sR_(sR), sN_(sN),
              geq_(geq), Tfield_(Tfield), drift_u_(drift_u) {}
        double Eval(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip) override {
            double Tval = 0.0;
            if (Tfield_) { Tfield_->GetValue(T, ip, Tval); }
            else { Tval = geq_.referenceTemperature; }
            const auto &om = dirs_.omega[idir_];
            const double gR = geq_.resistiveBE(Tval);
            const double gN = geq_.normalDriftedBE(Tval, om, drift_u_);
            return sR_*gR + sN_*gN;
        }
    private:
        const SNDirections& dirs_;
        int idir_;
        double sR_, sN_;
        GrayEquilibrium geq_;
        const mfem::GridFunction* Tfield_;
        std::array<double,3> drift_u_;
    };
    sys.bform = std::make_unique<mfem::LinearForm>(&fes);
    auto eq_coef = std::make_unique<EqCoefficient>(dirs, idir, sigmaR, sigmaN, geq, T_field, drift_u);
    sys.bform->AddDomainIntegrator(new mfem::DomainLFIntegrator(*eq_coef));
    // Keep EqCoefficient alive with the system
    sys.owned_coeffs.push_back(std::move(eq_coef));
    // Note: boundary integrators are added by the caller before Assemble().
    sys.solution = std::make_unique<mfem::GridFunction>(&fes);
    (*sys.solution) = 0.0;
    return sys;
}

inline void solveDirectionSystem(DirectionSystem& sys)
{
    mfem::Array<int> ess_tdof_list; // DG: no essential boundary dofs
    mfem::SparseMatrix A;
    mfem::Vector X, B;
    sys.Aform->FormLinearSystem(ess_tdof_list, *sys.solution, *sys.bform, A, X, B);

    mfem::GSSmoother M(A); // Gauss-Seidel often works better than Jacobi for advection
    mfem::GMRESSolver gmres;
    gmres.SetRelTol(1e-10);
    gmres.SetMaxIter(1000);
    gmres.SetKDim(100);
    gmres.SetPrintLevel(0);
    gmres.SetPreconditioner(M);
    gmres.SetOperator(A);
    gmres.Mult(B, X);

    sys.Aform->RecoverFEMSolution(X, *sys.bform, *sys.solution);
}

} // namespace dg

