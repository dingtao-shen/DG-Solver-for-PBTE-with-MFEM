#include "MacroscopicQuantities.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>

namespace pbte
{
namespace
{
mfem::Vector ComputeBBoxMin(const mfem::Mesh &mesh)
{
    const int sdim = mesh.SpaceDimension();
    mfem::Vector mn(sdim);
    for (int d = 0; d < sdim; ++d) { mn[d] = std::numeric_limits<double>::infinity(); }
    for (int v = 0; v < mesh.GetNV(); ++v)
    {
        const double *x = mesh.GetVertex(v);
        for (int d = 0; d < sdim; ++d) { mn[d] = std::min(mn[d], x[d]); }
    }
    return mn;
}

mfem::Vector ComputeBBoxMax(const mfem::Mesh &mesh)
{
    const int sdim = mesh.SpaceDimension();
    mfem::Vector mx(sdim);
    for (int d = 0; d < sdim; ++d) { mx[d] = -std::numeric_limits<double>::infinity(); }
    for (int v = 0; v < mesh.GetNV(); ++v)
    {
        const double *x = mesh.GetVertex(v);
        for (int d = 0; d < sdim; ++d) { mx[d] = std::max(mx[d], x[d]); }
    }
    return mx;
}

/// Brute-force point location: loop all elements, try inverse mapping.
/// Returns element id or -1 if not found.
int FindContainingElement2D(mfem::Mesh &mesh,
                            const mfem::Vector &pt,
                            mfem::IntegrationPoint &ip_out,
                            double tol = 1e-10)
{
    MFEM_VERIFY(mesh.Dimension() == 2, "FindContainingElement2D requires a 2D mesh.");
    const int ne = mesh.GetNE();
    for (int e = 0; e < ne; ++e)
    {
        mfem::ElementTransformation *tr = mesh.GetElementTransformation(e);
        mfem::InverseElementTransformation inv(tr);
        // inv.SetPrintLevel(0);
        inv.SetPrintLevel(-1);
        inv.SetMaxIter(64);
        inv.SetReferenceTol(tol);
        inv.SetPhysicalRelTol(tol);
        inv.SetInitialGuessType(mfem::InverseElementTransformation::ClosestPhysNode);
        inv.SetSolverType(mfem::InverseElementTransformation::NewtonElementProject);

        mfem::IntegrationPoint ip_ref;
        const int res = inv.Transform(pt, ip_ref);
        if (res == mfem::InverseElementTransformation::Inside)
        {
            ip_out = ip_ref;
            return e;
        }
    }
    return -1;
}
} // namespace

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
    const double factor = inv_kn * dir.weight * dw / heat_cap_v_;
    Tc_.Add(factor, coeff);

    // Heat flux components.
    const double vg = props_.GroupVelocity(branch)[spec];
    for (int d = 0; d < dim_; ++d)
    {
        const double flux_factor = factor * vg * dir.direction[d];
        Qc_[d].Add(flux_factor, coeff);
    }
}

void MacroscopicQuantities::Finalize(const std::vector<ElementIntegralData> &elem_data)
{

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
    const double denom = Tv_.Norml2();
    MFEM_VERIFY(denom > 0.0, "Tv norm is zero; cannot compute residual.");
    mfem::Vector diff(Tv_);
    diff -= prev_Tv;
    return diff.Norml2() / denom;
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

void MacroscopicQuantities::Write2DSliceTemperature(const std::string &out_path,
                                                    int Nx,
                                                    int Ny,
                                                    double clamp_tol) const
{
    MFEM_VERIFY(fes_.GetMesh() != nullptr, "FESpace has null mesh.");
    mfem::Mesh &mesh = *fes_.GetMesh();
    MFEM_VERIFY(mesh.SpaceDimension() == 2, "Write2DSliceTemperature currently supports 2D meshes only.");
    MFEM_VERIFY(Nx >= 2 && Ny >= 2, "Nx/Ny must be >= 2.");

    const mfem::Vector mn = ComputeBBoxMin(mesh);
    const mfem::Vector mx = ComputeBBoxMax(mesh);

    // Prepare output file.
    namespace fs = std::filesystem;
    fs::path p(out_path);
    if (p.has_parent_path())
    {
        fs::create_directories(p.parent_path());
    }
    std::ofstream ofs(p);
    MFEM_VERIFY(static_cast<bool>(ofs), "Failed to open output file for sampling: " + p.string());
    ofs.setf(std::ios::fixed);
    ofs.precision(16);
    ofs << "# nx " << Nx << " ny " << Ny << "\n";
    ofs << "x y T\n";

    const int sdim = mesh.SpaceDimension();
    mfem::Vector pt(sdim);
    mfem::IntegrationPoint ip;

    // Sampling grid in the bounding box.
    for (int j = 0; j < Ny; ++j)
    {
        const double tj = static_cast<double>(j) / static_cast<double>(Ny - 1);
        const double y = mn[1] + tj * (mx[1] - mn[1]);
        for (int i = 0; i < Nx; ++i)
        {
            const double ti = static_cast<double>(i) / static_cast<double>(Nx - 1);
            const double x = mn[0] + ti * (mx[0] - mn[0]);

            // Clamp slightly inside the domain to avoid ambiguous boundary inverse maps.
            double xc = x, yc = y;
            if (i == Nx - 1) { xc = mx[0] - clamp_tol; }
            if (j == Ny - 1) { yc = mx[1] - clamp_tol; }
            if (i == 0) { xc = mn[0] + clamp_tol; }
            if (j == 0) { yc = mn[1] + clamp_tol; }

            pt[0] = xc;
            pt[1] = yc;

            int elem = FindContainingElement2D(mesh, pt, ip);
            double T = std::numeric_limits<double>::quiet_NaN();
            if (elem >= 0)
            {
                const mfem::FiniteElement *fe = fes_.GetFE(elem);
                const int ndof = fe->GetDof();
                mfem::Vector shape(ndof);
                fe->CalcShape(ip, shape);
                double val = 0.0;
                for (int k = 0; k < ndof; ++k)
                {
                    val += Tc_(k, elem) * shape[k];
                }
                T = val;
            }

            ofs << x << " " << y << " " << T << "\n";
        }
    }
    ofs << std::flush;
    std::cout << "2D temperature slice written to: " << p << std::endl;
}

}  // namespace pbte


