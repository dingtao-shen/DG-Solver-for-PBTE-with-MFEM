#pragma once

#include "AngularQuadrature.hpp"
#include "ElementIntegrator.hpp"
#include "PhononProperties.hpp"

#include "mfem.hpp"

#include <memory>
#include <vector>

namespace pbte
{
/// Aggregate and compute macroscopic temperature/heat-flux fields from
/// directional (angle × branch × spectral) DG coefficients.
///
/// Storage is always “local” (serial: all elements; parallel: local elements
/// on this rank). No global gather is performed; callers can optionally
/// assemble/globalize with mfem::ParGridFunction or MPI collectives if needed.
class MacroscopicQuantities
{
public:
    MacroscopicQuantities(const mfem::FiniteElementSpace &fes,
                          const PhononProperties &props,
                          const AngleQuadrature &quad);

    /// Zero all accumulators before a new sweep/outer iteration.
    void Reset();

    /// Accumulate one directional coefficient block.
    /// coeff: ndof x ne_local matrix (columns = element-local DOFs).
    void AccumulateDirectionalCoeff(int dir_idx,
                                    int branch,
                                    int spec,
                                    const mfem::DenseMatrix &coeff);

    /// Finalize macroscopic fields after all directions/branches/specs are
    /// accumulated. Requires element-wise basis integrals from DGElementIntegrator.
    void Finalize(const std::vector<ElementIntegralData> &elem_data);

    /// Relative residual wrt previous average temperature vector.
    double Residual(const mfem::Vector &prev_Tv) const;

    /// Write high-order fields (Tc, Qc) to ParaView-friendly collection.
    /// prefix: file prefix; will emit .pvtu/.vtu (parallel) or .vtu (serial).
    /// high_order: if true, writes high-order L2; else exports as linearized.
    void WriteParaView(const std::string &prefix, bool high_order = true) const;

    // Accessors (local data).
    const mfem::DenseMatrix &Tc() const { return Tc_; }
    const mfem::Vector &Tv() const { return Tv_; }
    const std::vector<mfem::DenseMatrix> &Qc() const { return Qc_; }
    const std::vector<mfem::Vector> &Qv() const { return Qv_; }
    double HeatCapV() const { return heat_cap_v_; }

private:
    const mfem::FiniteElementSpace &fes_;
    const PhononProperties &props_;
    const AngleQuadrature &quad_;

    int dim_ = 0;
    int ndof_ = 0;
    int ne_ = 0;
    double heat_cap_v_ = 0.0;

    mfem::DenseMatrix Tc_;                 // ndof x ne_local
    mfem::Vector Tv_;                      // ne_local
    std::vector<mfem::DenseMatrix> Qc_;    // size=dim, each ndof x ne_local
    std::vector<mfem::Vector> Qv_;         // size=dim, each ne_local
};
}  // namespace pbte


