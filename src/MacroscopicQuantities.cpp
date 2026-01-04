#include "MacroscopicQuantities.hpp"

#include <cassert>
#include <iostream>

namespace pbte
{
MacroscopicQuantities::MacroscopicQuantities(const mfem::FiniteElementSpace &fes,
                                             const PhononProperties &props,
                                             const AngleQuadrature &quad)
    : fes_(fes), props_(props), quad_(quad)
{
    dim_ = fes_.GetMesh()->Dimension();
    ne_ = fes_.GetNE();
    // Assume uniform order L2 space; take ndof from element 0.
    ndof_ = fes_.GetFE(0)->GetDof();

    heat_cap_v_ = props_.avgHeatCapacity();

    Tc_.SetSize(ndof_, ne_);
    Tv_.SetSize(ne_);
    Qc_.assign(dim_, mfem::DenseMatrix(ndof_, ne_));
    Qv_.assign(dim_, mfem::Vector(ne_));

    Reset();
}

void MacroscopicQuantities::Reset()
{
    Tc_ = 0.0;
    Tv_ = 0.0;
    for (int d = 0; d < dim_; ++d)
    {
        Qc_[d] = 0.0;
        Qv_[d] = 0.0;
    }
}

void MacroscopicQuantities::AccumulateDirectionalCoeff(int dir_idx,
                                                       int branch,
                                                       int spec,
                                                       const mfem::DenseMatrix &coeff)
{
    assert(coeff.Height() == ndof_);
    assert(coeff.Width() == ne_);
    const auto &dirs = quad_.Directions();
    assert(dir_idx >= 0 && dir_idx < static_cast<int>(dirs.size()));
    const auto &dir = dirs[dir_idx];

    // Scalar weight factor for temperature accumulation.
    const double inv_kn = props_.InvKn(branch)[spec];
    const double dw = props_.FrequencyWeight(branch)[spec];
    const double factor = inv_kn * dir.weight * dw;

    Tc_.Add(factor, coeff);

    // Heat flux components.
    const double vg = props_.GroupVelocity(branch)[spec];
    for (int d = 0; d < dim_; ++d)
    {
        const double flux_factor = factor * vg * dir.direction[d];
        Qc_[d].Add(flux_factor, coeff);
    }

    // One-time debug to ensure accumulation is nonzero.
    static bool printed = false;
    if (!printed)
    {
        printed = true;
        std::cout << "[Macro dbg] dir_idx=" << dir_idx
                  << " branch=" << branch
                  << " spec=" << spec
                  << " inv_kn=" << inv_kn
                  << " dw=" << dw
                  << " dir_w=" << dir.weight
                  << " factor=" << factor
                  << " coeff_F=" << coeff.FNorm()
                  << std::endl;
    }
}

void MacroscopicQuantities::Finalize(const std::vector<ElementIntegralData> &elem_data)
{
    // const double scale = (heat_cap_v_ != 0.0) ? (1.0 / heat_cap_v_) : 0.0;
    std::cout << "[Macro dbg] Tc_FNorm_before_scale=" << Tc_.FNorm() << std::endl;
    const double scale = 1.0 / heat_cap_v_;
    Tc_ *= scale;

    static bool printed = false;
    if (!printed)
    {
        printed = true;
        std::cout << "[Macro dbg] Tc_FNorm_after_scale=" << Tc_.FNorm()
                  << " scale=" << scale << std::endl;
    }

    // Compute cell-average temperature and heat flux using basis integrals.
    for (int e = 0; e < ne_; ++e)
    {
        const mfem::Vector &m = elem_data[e].basis_integrals;
        assert(m.Size() == ndof_);

        // Tv
        Tv_[e] = 0.0;
        for (int i = 0; i < ndof_; ++i)
        {
            Tv_[e] += Tc_(i, e) * m[i];
        }

        // Qv
        for (int d = 0; d < dim_; ++d)
        {
            double q = 0.0;
            for (int i = 0; i < ndof_; ++i)
            {
                q += Qc_[d](i, e) * m[i];
            }
            Qv_[d][e] = q;
        }
    }
}

double MacroscopicQuantities::Residual(const mfem::Vector &prev_Tv) const
{
    mfem::Vector diff = Tv_;
    diff -= prev_Tv;
    const double num = diff.Norml2();
    const double den = Tv_.Norml2();
    if (den == 0.0)
    {
        return num; // fall back to absolute change if new Tv is zero
    }
    return num / den;
}

void MacroscopicQuantities::WriteParaView(const std::string &prefix, bool high_order) const
{
    // Scalar field (Tc) on original space.
    std::unique_ptr<mfem::GridFunction> Tc_gf;
    std::unique_ptr<mfem::GridFunction> Q_gf;

    const mfem::FiniteElementCollection *fec = fes_.FEColl();
    MFEM_VERIFY(fec != nullptr, "FEColl is null");

    const auto ordering = fes_.GetOrdering();

    // Helper to pack Q element vector respecting ordering.
    auto pack_q_elem = [&](int e, mfem::Vector &loc) {
        loc.SetSize(dim_ * ndof_);
        for (int i = 0; i < ndof_; ++i)
        {
            for (int d = 0; d < dim_; ++d)
            {
                int idx = (ordering == mfem::Ordering::byNODES)
                              ? i * dim_ + d
                              : d * ndof_ + i;
                loc[idx] = Qc_[d](i, e);
            }
        }
    };

#ifdef MFEM_USE_MPI
    if (auto pfes = dynamic_cast<const mfem::ParFiniteElementSpace *>(&fes_))
    {
        auto pmesh = pfes->GetParMesh();
        // Scalar Tc
        Tc_gf = std::make_unique<mfem::ParGridFunction>(
            const_cast<mfem::ParFiniteElementSpace *>(pfes));
        (*Tc_gf) = 0.0;

        // Vector Q
        auto vfes = std::make_unique<mfem::ParFiniteElementSpace>(
            pmesh, fec, dim_, ordering);
        Q_gf = std::make_unique<mfem::ParGridFunction>(vfes.get());
        (*Q_gf) = 0.0;

        mfem::Array<int> vdofs;
        mfem::Vector loc;
        for (int e = 0; e < ne_; ++e)
        {
            pfes->GetElementVDofs(e, vdofs);
            // Tc
            for (int j = 0; j < vdofs.Size(); ++j)
            {
                (*Tc_gf)(vdofs[j]) = Tc_(j, e);
            }
            // Q
            vfes->GetElementVDofs(e, vdofs);
            pack_q_elem(e, loc);
            Q_gf->SetSubVector(vdofs, loc);
        }

        mfem::ParaViewDataCollection dc(prefix, pmesh);
        dc.SetPrefixPath("output/vis");
        dc.SetDataFormat(mfem::VTKFormat::BINARY);
        dc.SetHighOrderOutput(high_order);
        dc.SetLevelsOfDetail(1);
        dc.RegisterField("Tc", Tc_gf.get());
        dc.RegisterField("Q", Q_gf.get());
        dc.Save();
        return;
    }
#endif

    // Serial path.
    {
        auto mesh = fes_.GetMesh();
        Tc_gf = std::make_unique<mfem::GridFunction>(
            const_cast<mfem::FiniteElementSpace *>(&fes_));
        (*Tc_gf) = 0.0;

        auto vfes = std::make_unique<mfem::FiniteElementSpace>(
            mesh, fec, dim_, ordering);
        Q_gf = std::make_unique<mfem::GridFunction>(vfes.get());
        (*Q_gf) = 0.0;

        mfem::Array<int> vdofs;
        mfem::Vector loc;
        for (int e = 0; e < ne_; ++e)
        {
            fes_.GetElementVDofs(e, vdofs);
            for (int j = 0; j < vdofs.Size(); ++j)
            {
                (*Tc_gf)(vdofs[j]) = Tc_(j, e);
            }
            vfes->GetElementVDofs(e, vdofs);
            pack_q_elem(e, loc);
            Q_gf->SetSubVector(vdofs, loc);
        }

        mfem::ParaViewDataCollection dc(prefix, mesh);
        dc.SetPrefixPath("output/vis");
        dc.SetDataFormat(mfem::VTKFormat::BINARY);
        dc.SetHighOrderOutput(high_order);
        dc.SetLevelsOfDetail(1);
        dc.RegisterField("Tc", Tc_gf.get());
        dc.RegisterField("Q", Q_gf.get());
        dc.Save();
    }
}
}  // namespace pbte


