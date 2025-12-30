#include "PBTESolver.hpp"

#include <cmath>
#include <memory>

namespace pbte
{
PBTESolver::PBTESolver(const mfem::Mesh &mesh,
                       const mfem::FiniteElementSpace &fes,
                       const AngleQuadrature &angle_quad,
                       const AngularSweepOrder &sweep_order,
                       const std::vector<ElementIntegralData> &elem_data,
                       const PhononProperties &props,
                       const std::map<int, double> &isothermal_bcs,
                       CachePolicy policy,
                       double tol,
                       int max_iter)
    : mesh_(mesh),
      fes_(fes),
      quad_(angle_quad),
      sweep_(sweep_order),
      elem_data_(elem_data),
      props_(props),
      iso_bc_(isothermal_bcs),
      policy_(policy),
      tol_(tol),
      max_iter_(max_iter)
{
    dim_ = mesh_.Dimension();
    ne_ = mesh_.GetNE();
    ndof_ = fes_.GetFE(0)->GetDof();
    ndir_ = static_cast<int>(quad_.Directions().size());
    nbranch_ = static_cast<int>(props_.Frequency().size());
    nspec_ = nbranch_ > 0 ? static_cast<int>(props_.Frequency(0).size()) : 0;
    omega_ = quad_.TotalWeight();

    dt_inv_ = 0.0;
    for (int b = 0; b < nbranch_; ++b)
    {
        for (int s = 0; s < nspec_; ++s)
        {
            dt_inv_ = std::max(dt_inv_, props_.InvKn(b)[s]);
        }
    }

    BuildCellData();

    if (policy_ == CachePolicy::FullLU)
    {
        lu_cache_.resize(ndir_);
        for (int k = 0; k < ndir_; ++k)
        {
            lu_cache_[k].resize(nbranch_);
            for (int b = 0; b < nbranch_; ++b)
            {
                lu_cache_[k][b].resize(nspec_);
                for (int s = 0; s < nspec_; ++s)
                {
                    LUBlock blk;
                    blk.A.resize(ne_);
                    blk.lu.resize(ne_);
                    for (int e = 0; e < ne_; ++e)
                    {
                        blk.A[e] = AssembleA(k, b, s, e);
                        blk.lu[e].Factor(blk.A[e]);
                    }
                    lu_cache_[k][b][s] = std::move(blk);
                }
            }
        }
    }
}

void PBTESolver::BuildCellData()
{
    cells_.resize(ne_);
    mfem::Vector dummy_normal;
    for (int e = 0; e < ne_; ++e)
    {
        CellData cd;
        cd.mass = elem_data_[e].mass_matrix;
        cd.mass_t = cd.mass;
        cd.mass_t.Transpose();
        cd.stiff = elem_data_[e].stiffness_matrices;
        const auto &fcs = elem_data_[e].face_couplings;
        for (const auto &fc : fcs)
        {
            FaceInfo fi;
            fi.face_id = fc.face_id;
            fi.is_boundary = (fc.neighbor_elem < 0 && !fc.is_shared);
            fi.boundary_attr = fc.boundary_attr;
            fi.neighbor_elem = fc.neighbor_elem;
            fi.normal = FaceNormal(fc.face_id, e);
            fi.coupling = fc.coupling;
            if (fc.neighbor_elem < 0)
            {
                fi.face_integral = fc.isothermal_rhs;
            }
            // face_mass_matrices aligned with face order
            if (cd.faces.size() < elem_data_[e].face_mass_matrices.size())
            {
                // push after fill below
            }
            cd.faces.push_back(std::move(fi));
        }
        // attach face_mass from face_mass_matrices (same order as face_couplings)
        for (size_t i = 0; i < cd.faces.size(); ++i)
        {
            if (i < elem_data_[e].face_mass_matrices.size())
            {
                cd.faces[i].face_mass = elem_data_[e].face_mass_matrices[i];
            }
        }
        cells_[e] = std::move(cd);
    }
}

mfem::Vector PBTESolver::FaceNormal(int face_id, int elem_id) const
{
    // Approximate face normal using vertices; orient outward from elem center.
    mfem::Array<int> verts;
    mesh_.GetFaceVertices(face_id, verts);
    const int sdim = mesh_.SpaceDimension();
    mfem::Vector n(sdim);
    n = 0.0;
    if (sdim == 2 && verts.Size() >= 2)
    {
        const double *v0 = mesh_.GetVertex(verts[0]);
        const double *v1 = mesh_.GetVertex(verts[1]);
        const double dx = v1[0] - v0[0];
        const double dy = v1[1] - v0[1];
        n[0] = dy;
        n[1] = -dx;
    }
    else if (sdim == 3 && verts.Size() >= 3)
    {
        const double *v0 = mesh_.GetVertex(verts[0]);
        const double *v1 = mesh_.GetVertex(verts[1]);
        const double *v2 = mesh_.GetVertex(verts[2]);
        mfem::Vector e1(3), e2(3);
        e1[0] = v1[0] - v0[0];
        e1[1] = v1[1] - v0[1];
        e1[2] = v1[2] - v0[2];
        e2[0] = v2[0] - v0[0];
        e2[1] = v2[1] - v0[1];
        e2[2] = v2[2] - v0[2];
        mfem::Vector c(3);
        c[0] = e1[1] * e2[2] - e1[2] * e2[1];
        c[1] = e1[2] * e2[0] - e1[0] * e2[2];
        c[2] = e1[0] * e2[1] - e1[1] * e2[0];
        n = c;
    }
    double norm = n.Norml2();
    if (norm > 0.0)
    {
        n /= norm;
    }
    // Orient outward: compare to vector from element centroid to face centroid.
    mfem::Vector elem_c(sdim), face_c(sdim);
    elem_c = 0.0;
    face_c = 0.0;
    mfem::Array<int> e_verts;
    mesh_.GetElementVertices(elem_id, e_verts);
    for (int v : e_verts)
    {
        const double *pv = mesh_.GetVertex(v);
        for (int d = 0; d < sdim; ++d) elem_c[d] += pv[d];
    }
    if (e_verts.Size() > 0) elem_c /= static_cast<double>(e_verts.Size());
    for (int v : verts)
    {
        const double *pv = mesh_.GetVertex(v);
        for (int d = 0; d < sdim; ++d) face_c[d] += pv[d];
    }
    if (verts.Size() > 0) face_c /= static_cast<double>(verts.Size());
    mfem::Vector to_face = face_c;
    to_face -= elem_c;
    if (n * to_face < 0.0)
    {
        n *= -1.0;
    }
    return n;
}

mfem::DenseMatrix PBTESolver::AssembleA(int dir_idx, int branch, int spec, int elem) const
{
    const auto &cd = cells_[elem];
    const auto &dir = quad_.Directions()[dir_idx];
    const double vg = props_.GroupVelocity(branch)[spec];
    mfem::DenseMatrix A(cd.mass);
    A *= dt_inv_;
    for (int d = 0; d < dim_; ++d)
    {
        A.Add(-vg * dir.direction[d], cd.stiff[d]); 
        // 注意这里有问题，stiff的dim是空间物理网格，dir.direction的维度是立体角离散维度，目前都为3没有问题，但要考虑不匹配时的情况
    }
    // faces
    for (const auto &f : cd.faces)
    {
        const double f_dot = f.normal * mfem::Vector(const_cast<double *>(dir.direction.data()), dim_);
        if (f_dot > 0.0)
        {
            const double coeff = vg * f_dot;
            A.Add(coeff, f.face_mass);
            // const double coeff = 0.5 * vg * (f_dot + std::abs(f_dot));
            // if (coeff != 0.0 && f.face_mass.Height() == ndof_)
            // {
            //     A.Add(coeff, f.face_mass);
            // }
        }
    }
    return A;
}

void PBTESolver::EnsureLU(int dir_idx, int branch, int spec, int elem)
{
    if (policy_ == CachePolicy::FullLU) return;
    if (dir_idx >= static_cast<int>(lu_cache_.size()))
    {
        lu_cache_.resize(ndir_);
    }
    if (lu_cache_[dir_idx].empty())
    {
        lu_cache_[dir_idx].resize(nbranch_);
    }
    if (lu_cache_[dir_idx][branch].empty())
    {
        lu_cache_[dir_idx][branch].resize(nspec_);
    }
    if (lu_cache_[dir_idx][branch][spec].A.empty())
    {
        lu_cache_[dir_idx][branch][spec].A.resize(ne_);
        lu_cache_[dir_idx][branch][spec].lu.resize(ne_);
    }
    auto &blk = lu_cache_[dir_idx][branch][spec];
    blk.A[elem] = AssembleA(dir_idx, branch, spec, elem);
    blk.lu[elem].Factor(blk.A[elem]);
}

void PBTESolver::LUBlock::LUSolver::Factor(const mfem::DenseMatrix &src)
{
    A = src;
    inv = std::make_unique<mfem::DenseMatrixInverse>(&A);
    inv->Factor();
}

void PBTESolver::LUBlock::LUSolver::Solve(const mfem::Vector &b, mfem::Vector &x) const
{
    MFEM_VERIFY(inv != nullptr, "LU not factored");
    inv->Mult(b, x);
}

double PBTESolver::Solve(std::vector<std::vector<std::vector<mfem::DenseMatrix>>> &coeff,
                         MacroscopicQuantities &macro)
{
    MFEM_VERIFY(static_cast<int>(coeff.size()) == ndir_, "coeff size mismatch (dir)");
    MFEM_VERIFY(static_cast<int>(coeff[0].size()) == nbranch_, "coeff size mismatch (branch)");
    MFEM_VERIFY(static_cast<int>(coeff[0][0].size()) == nspec_, "coeff size mismatch (spec)");

    mfem::Vector prev_Tv = macro.Tv();  // initial (all zeros)

    for (int iter = 0; iter < max_iter_; ++iter)
    {
        macro.Reset();
        double bndry_rhs_acc = 0.0;
        double inflow_acc = 0.0;

        for (int k = 0; k < ndir_; ++k)
        {
            const auto &order = sweep_.Order(k);
            for (int b = 0; b < nbranch_; ++b)
            {
                for (int s = 0; s < nspec_; ++s)
                {
                    const double invKn = props_.InvKn(b)[s];
                    const double Cwp = props_.HeatCapacity(b)[s];
                    const double vg = props_.GroupVelocity(b)[s];
                    const auto &dir = quad_.Directions()[k];
                    mfem::Vector dir_vec(dim_);
                    for (int d = 0; d < dim_; ++d) dir_vec[d] = dir.direction[d];

                    auto &coeff_mat = coeff[k][b][s];
                    // sanity
                    if (coeff_mat.Height() != ndof_ || coeff_mat.Width() != ne_)
                    {
                        coeff_mat.SetSize(ndof_, ne_);
                        coeff_mat = 0.0;
                    }

                    mfem::Vector rhs(ndof_);
                    mfem::Vector sol(ndof_);

                    for (int idx = 0; idx < static_cast<int>(order.size()); ++idx)
                    {
                        const int e = order[idx];
                        const auto &cd = cells_[e];

                        // Assemble rhs
                        rhs = 0.0;
                        // invKn*Cwp/Ω * M^T * Tc
                        mfem::Vector tmp(ndof_);
                        {
                            mfem::Vector tc_col(tmp.GetData(), ndof_);
                            macro.Tc().GetColumn(e, tc_col);
                            cd.mass_t.Mult(tc_col, tmp);
                            rhs.Add(invKn * Cwp / omega_, tmp);
                        }
                        // (dt_inv - invKn) * M^T * u_old
                        {
                            mfem::Vector u_col(tmp.GetData(), ndof_);
                            coeff_mat.GetColumn(e, u_col);
                            cd.mass_t.Mult(u_col, tmp);
                            rhs.Add(dt_inv_ - invKn, tmp);
                        }

                        // boundary / neighbor terms
                        double max_coeff_in_dbg = 0.0;
                        for (const auto &f : cd.faces)
                        {
                            const double f_dot = f.normal * dir_vec;
                            const double coeff_in = 0.5 * vg * (f_dot - std::abs(f_dot));
                                max_coeff_in_dbg = std::max(max_coeff_in_dbg, std::abs(coeff_in));

                            if (f.is_boundary)
                            {
                                if (coeff_in != 0.0 && iso_bc_.count(f.boundary_attr))
                                {
                                    const double Tbc = iso_bc_.at(f.boundary_attr);
                                    rhs.Add(-coeff_in * Cwp / omega_ * Tbc, f.face_integral);
                                    bndry_rhs_acc += std::abs(coeff_in * Cwp / omega_ * Tbc) * f.face_integral.Norml2();
                                }
                            }
                            else
                            {
                                // interior neighbor
                                const int nbr = f.neighbor_elem;
                                if (nbr >= 0)
                                {
                                    if (coeff_in != 0.0 && f.coupling.Height() == ndof_)
                                    {
                                        mfem::Vector u_nbr(tmp.GetData(), ndof_);
                                        coeff[k][b][s].GetColumn(nbr, u_nbr);
                                        f.coupling.Mult(u_nbr, tmp);
                                        rhs.Add(-coeff_in, tmp);
                                    }
                                }
                            }
                        }

                            if (iter == 0 && k == 0 && b == 0 && s == 0 && e == 0)
                            {
                                std::cout << "[Serial dbg] rhs_norm=" << rhs.Norml2()
                                          << " max_coeff_in=" << max_coeff_in_dbg << std::endl;
                            }

                        // Solve
                        EnsureLU(k, b, s, e);
                        const auto &lu = lu_cache_[k][b][s].lu[e];
                        mfem::Vector rhs_copy(rhs);
                        lu.Solve(rhs_copy, sol);
                        coeff_mat.SetCol(e, sol);

                        if (iter == 0 && k == 0 && b == 0 && s == 0 && e == 0)
                        {
                            std::cout << "[Serial dbg] first sol norm=" << sol.Norml2() << std::endl;
                        }
                    }  // cells

                    // accumulate macro for this (dir,b,s)
                    macro.AccumulateDirectionalCoeff(k, b, s, coeff_mat);
                    inflow_acc = std::max(inflow_acc, bndry_rhs_acc);
                }
            }
        }  // dirs

        macro.Finalize(elem_data_);
        const double res = macro.Residual(prev_Tv);
        if (iter == 0 || res > tol_)
        {
            std::cout.setf(std::ios::scientific);
            std::cout.precision(6);
            std::cout << "[Serial] iter " << (iter + 1) << ", residual = " << res << std::endl;
            std::cout << "[Serial dbg] boundary_rhs_acc=" << inflow_acc
                      << " Tv_norm=" << macro.Tv().Norml2()
                      << " Tc_norm=" << macro.Tc().FNorm() << std::endl;
            std::cout.unsetf(std::ios::scientific);
        }
        if (res < tol_) return res;
        prev_Tv = macro.Tv();
    }
    return macro.Residual(prev_Tv);
}

}  // namespace pbte

#ifdef MFEM_USE_MPI

namespace pbte
{

namespace
{
void CopyColumn(const mfem::DenseMatrix &m, int col, mfem::Vector &out)
{
    out.SetSize(m.Height());
    m.GetColumn(col, out);
}
}  // namespace

PBTESolverPar::PBTESolverPar(const mfem::ParMesh &pmesh,
                             const mfem::ParFiniteElementSpace &pfes,
                             const AngleQuadrature &angle_quad,
                             const AngularSweepOrder &sweep_order,
                             const std::vector<ElementIntegralData> &elem_data,
                             const PhononProperties &props,
                             const std::map<int, double> &isothermal_bcs,
                             CachePolicy policy,
                             double tol,
                             int max_iter)
    : pmesh_(pmesh),
      pfes_(pfes),
      quad_(angle_quad),
      sweep_(sweep_order),
      elem_data_(elem_data),
      props_(props),
      iso_bc_(isothermal_bcs),
      policy_(policy),
      tol_(tol),
      max_iter_(max_iter)
{
    dim_ = pmesh_.Dimension();
    ne_ = pmesh_.GetNE();  // local elements
    ndof_ = pfes_.GetFE(0)->GetDof();
    ndir_ = static_cast<int>(quad_.Directions().size());
    nbranch_ = static_cast<int>(props_.Frequency().size());
    nspec_ = nbranch_ > 0 ? static_cast<int>(props_.Frequency(0).size()) : 0;
    omega_ = quad_.TotalWeight();

    dt_inv_ = 0.0;
    for (int b = 0; b < nbranch_; ++b)
    {
        for (int s = 0; s < nspec_; ++s)
        {
            dt_inv_ = std::max(dt_inv_, props_.InvKn(b)[s]);
        }
    }

    // Map face_id -> shared face index
    const int nshf = pmesh_.GetNSharedFaces();
    for (int sf = 0; sf < nshf; ++sf)
    {
        const int fid = pmesh_.GetSharedFace(sf);
        face_to_shared_idx_[fid] = sf;
    }

    BuildCellData();

    if (policy_ == CachePolicy::FullLU)
    {
        lu_cache_.resize(ndir_);
        for (int k = 0; k < ndir_; ++k)
        {
            lu_cache_[k].resize(nbranch_);
            for (int b = 0; b < nbranch_; ++b)
            {
                lu_cache_[k][b].resize(nspec_);
                for (int s = 0; s < nspec_; ++s)
                {
                    LUBlock blk;
                    blk.A.resize(ne_);
                    blk.lu.resize(ne_);
                    for (int e = 0; e < ne_; ++e)
                    {
                        blk.A[e] = AssembleA(k, b, s, e);
                        blk.lu[e].Factor(blk.A[e]);
                    }
                    lu_cache_[k][b][s] = std::move(blk);
                }
            }
        }
    }
}

void PBTESolverPar::LUBlock::LUSolver::Factor(const mfem::DenseMatrix &src)
{
    A = src;
    inv = std::make_unique<mfem::DenseMatrixInverse>(&A);
    inv->Factor();
}

void PBTESolverPar::LUBlock::LUSolver::Solve(const mfem::Vector &b, mfem::Vector &x) const
{
    MFEM_VERIFY(inv != nullptr, "LU not factored");
    inv->Mult(b, x);
}

mfem::Vector PBTESolverPar::FaceNormal(int face_id, int elem_id) const
{
    mfem::Array<int> verts;
    pmesh_.GetFaceVertices(face_id, verts);
    const int sdim = pmesh_.SpaceDimension();
    mfem::Vector n(sdim);
    n = 0.0;
    if (sdim == 2 && verts.Size() >= 2)
    {
        const double *v0 = pmesh_.GetVertex(verts[0]);
        const double *v1 = pmesh_.GetVertex(verts[1]);
        const double dx = v1[0] - v0[0];
        const double dy = v1[1] - v0[1];
        n[0] = dy;
        n[1] = -dx;
    }
    else if (sdim == 3 && verts.Size() >= 3)
    {
        const double *v0 = pmesh_.GetVertex(verts[0]);
        const double *v1 = pmesh_.GetVertex(verts[1]);
        const double *v2 = pmesh_.GetVertex(verts[2]);
        mfem::Vector e1(3), e2(3);
        e1[0] = v1[0] - v0[0];
        e1[1] = v1[1] - v0[1];
        e1[2] = v1[2] - v0[2];
        e2[0] = v2[0] - v0[0];
        e2[1] = v2[1] - v0[1];
        e2[2] = v2[2] - v0[2];
        mfem::Vector c(3);
        c[0] = e1[1] * e2[2] - e1[2] * e2[1];
        c[1] = e1[2] * e2[0] - e1[0] * e2[2];
        c[2] = e1[0] * e2[1] - e1[1] * e2[0];
        n = c;
    }
    double norm = n.Norml2();
    if (norm > 0.0) n /= norm;

    mfem::Vector elem_c(sdim), face_c(sdim);
    elem_c = 0.0;
    face_c = 0.0;
    mfem::Array<int> e_verts;
    pmesh_.GetElementVertices(elem_id, e_verts);
    for (int v : e_verts)
    {
        const double *pv = pmesh_.GetVertex(v);
        for (int d = 0; d < sdim; ++d) elem_c[d] += pv[d];
    }
    if (e_verts.Size() > 0) elem_c /= static_cast<double>(e_verts.Size());
    for (int v : verts)
    {
        const double *pv = pmesh_.GetVertex(v);
        for (int d = 0; d < sdim; ++d) face_c[d] += pv[d];
    }
    if (verts.Size() > 0) face_c /= static_cast<double>(verts.Size());
    mfem::Vector to_face = face_c;
    to_face -= elem_c;
    if (n * to_face < 0.0) n *= -1.0;
    return n;
}

void PBTESolverPar::BuildCellData()
{
    cells_.resize(ne_);
    for (int e = 0; e < ne_; ++e)
    {
        CellData cd;
        cd.mass = elem_data_[e].mass_matrix;
        cd.mass_t = cd.mass;
        cd.mass_t.Transpose();
        cd.stiff = elem_data_[e].stiffness_matrices;

        const auto &fcs = elem_data_[e].face_couplings;
        for (size_t i = 0; i < fcs.size(); ++i)
        {
            const auto &fc = fcs[i];
            FaceInfo fi;
            fi.face_id = fc.face_id;
            fi.is_boundary = (fc.neighbor_elem < 0 && !fc.is_shared);
            fi.boundary_attr = fc.boundary_attr;
            fi.neighbor_elem = fc.neighbor_elem;
            fi.normal = FaceNormal(fc.face_id, e);
            fi.coupling = fc.coupling;
            fi.is_shared = fc.is_shared;
            if (fc.neighbor_elem < 0 && !fc.is_shared)
            {
                fi.face_integral = fc.isothermal_rhs;
            }
            if (i < elem_data_[e].face_mass_matrices.size())
            {
                fi.face_mass = elem_data_[e].face_mass_matrices[i];
            }
            if (fc.is_shared)
            {
                const int sf = face_to_shared_idx_.at(fc.face_id);
                pfes_.GetFaceNbrElementVDofs(sf, fi.face_nbr_vdofs);
            }
            cd.faces.push_back(std::move(fi));
        }
        cells_[e] = std::move(cd);
    }
}

mfem::DenseMatrix PBTESolverPar::AssembleA(int dir_idx, int branch, int spec, int elem) const
{
    const auto &cd = cells_[elem];
    const auto &dir = quad_.Directions()[dir_idx];
    const double vg = props_.GroupVelocity(branch)[spec];
    mfem::DenseMatrix A(cd.mass);
    A *= dt_inv_;
    for (int d = 0; d < dim_; ++d)
    {
        A.Add(-vg * dir.direction[d], cd.stiff[d]);
    }
    for (const auto &f : cd.faces)
    {
        const double f_dot = f.normal * mfem::Vector(const_cast<double *>(dir.direction.data()), dim_);
        if (f_dot > 0.0)
        {
            const double coeff = 0.5 * vg * (f_dot + std::abs(f_dot));
            if (coeff != 0.0 && f.face_mass.Height() == ndof_)
            {
                A.Add(coeff, f.face_mass);
            }
        }
    }
    return A;
}

void PBTESolverPar::EnsureLU(int dir_idx, int branch, int spec, int elem)
{
    if (policy_ == CachePolicy::FullLU) return;
    if (dir_idx >= static_cast<int>(lu_cache_.size()))
    {
        lu_cache_.resize(ndir_);
    }
    if (lu_cache_[dir_idx].empty())
    {
        lu_cache_[dir_idx].resize(nbranch_);
    }
    if (lu_cache_[dir_idx][branch].empty())
    {
        lu_cache_[dir_idx][branch].resize(nspec_);
    }
    if (lu_cache_[dir_idx][branch][spec].A.empty())
    {
        lu_cache_[dir_idx][branch][spec].A.resize(ne_);
        lu_cache_[dir_idx][branch][spec].lu.resize(ne_);
    }
    auto &blk = lu_cache_[dir_idx][branch][spec];
    blk.A[elem] = AssembleA(dir_idx, branch, spec, elem);
    blk.lu[elem].Factor(blk.A[elem]);
}

double PBTESolverPar::Solve(std::vector<std::vector<std::vector<mfem::DenseMatrix>>> &coeff,
                            MacroscopicQuantities &macro)
{
    MFEM_VERIFY(static_cast<int>(coeff.size()) == ndir_, "coeff size mismatch (dir)");
    MFEM_VERIFY(static_cast<int>(coeff[0].size()) == nbranch_, "coeff size mismatch (branch)");
    MFEM_VERIFY(static_cast<int>(coeff[0][0].size()) == nspec_, "coeff size mismatch (spec)");

    mfem::Vector prev_Tv = macro.Tv();

    // temporary buffers
    mfem::Vector tmp(ndof_), u_col(ndof_), u_nbr(ndof_), rhs(ndof_), sol(ndof_);
    mfem::Vector dir_vec(dim_);

    for (int iter = 0; iter < max_iter_; ++iter)
    {
        macro.Reset();

        for (int k = 0; k < ndir_; ++k)
        {
            const auto &order = sweep_.Order(k);  // local element order
            double bndry_rhs_acc = 0.0;
            for (int b = 0; b < nbranch_; ++b)
            {
                for (int s = 0; s < nspec_; ++s)
                {
                    const double invKn = props_.InvKn(b)[s];
                    const double Cwp = props_.HeatCapacity(b)[s];
                    const double vg = props_.GroupVelocity(b)[s];
                    const auto &dir = quad_.Directions()[k];
                    for (int d = 0; d < dim_; ++d) dir_vec[d] = dir.direction[d];

                    // ParGridFunction to access face neighbor dofs
                    mfem::ParGridFunction gf(const_cast<mfem::ParFiniteElementSpace *>(&pfes_));
                    gf = 0.0;
                    // load local coeff into gf
                    mfem::Array<int> vdofs;
                    for (int e = 0; e < ne_; ++e)
                    {
                        pfes_.GetElementVDofs(e, vdofs);
                        CopyColumn(coeff[k][b][s], e, u_col);
                        gf.SetSubVector(vdofs, u_col);
                    }
                    gf.ExchangeFaceNbrData();

                    for (int idx = 0; idx < static_cast<int>(order.size()); ++idx)
                    {
                        const int e = order[idx];
                        const auto &cd = cells_[e];

                        // rhs = invKn*Cwp/Ω * M^T * Tc + (dt_inv - invKn)*M^T*u_old
                        macro.Tc().GetColumn(e, u_col);
                        cd.mass_t.Mult(u_col, tmp);
                        rhs = 0.0;
                        rhs.Add(invKn * Cwp / omega_, tmp);

                        CopyColumn(coeff[k][b][s], e, u_col);
                        cd.mass_t.Mult(u_col, tmp);
                        rhs.Add(dt_inv_ - invKn, tmp);

                        double max_coeff_in_dbg = 0.0;
                        for (const auto &f : cd.faces)
                        {
                            const double f_dot = f.normal * dir_vec;
                            const double coeff_in = 0.5 * vg * (f_dot - std::abs(f_dot));
                            max_coeff_in_dbg = std::max(max_coeff_in_dbg, std::abs(coeff_in));

                            if (f.is_boundary)
                            {
                                if (coeff_in != 0.0 && iso_bc_.count(f.boundary_attr))
                                {
                                    const double Tbc = iso_bc_.at(f.boundary_attr);
                                    rhs.Add(-coeff_in * Cwp / omega_ * Tbc, f.face_integral);
                                    if (iter == 0 && k == 0 && b == 0 && s == 0 && e == 0)
                                    {
                                        std::cout << "[Parallel dbg] boundary inflow: coeff_in="
                                                  << coeff_in
                                                  << " attr=" << f.boundary_attr
                                                  << " |face_int|=" << f.face_integral.Norml2()
                                                  << " Cwp=" << Cwp
                                                  << " omega=" << omega_
                                                  << " Tbc=" << Tbc
                                                  << std::endl;
                                    }
                                    bndry_rhs_acc += std::abs(coeff_in * Cwp / omega_ * Tbc) * f.face_integral.Norml2();
                                }
                            }
                            else
                            {
                                if (f.is_shared)
                                {
                                    if (coeff_in != 0.0 && f.coupling.Height() == ndof_)
                                    {
                                        gf.GetSubVector(f.face_nbr_vdofs, u_nbr);
                                        f.coupling.Mult(u_nbr, tmp);
                                        rhs.Add(-coeff_in, tmp);
                                    }
                                }
                                else
                                {
                                    const int nbr = f.neighbor_elem;
                                    if (nbr >= 0 && coeff_in != 0.0 && f.coupling.Height() == ndof_)
                                    {
                                        CopyColumn(coeff[k][b][s], nbr, u_nbr);
                                        f.coupling.Mult(u_nbr, tmp);
                                        rhs.Add(-coeff_in, tmp);
                                    }
                                }
                            }
                        }

                        if (iter == 0 && k == 0 && b == 0 && s == 0 && e == 0)
                        {
                            std::cout << "[Parallel dbg] rhs_norm=" << rhs.Norml2()
                                      << " max_coeff_in=" << max_coeff_in_dbg << std::endl;
                        }

                        EnsureLU(k, b, s, e);
                        const auto &lu = lu_cache_[k][b][s].lu[e];
                        lu.Solve(rhs, sol);
                        coeff[k][b][s].SetCol(e, sol);
                    }

                    // accumulate macro
                    macro.AccumulateDirectionalCoeff(k, b, s, coeff[k][b][s]);
                }
            }
        }

        macro.Finalize(elem_data_);
        const double res = macro.Residual(prev_Tv);
        std::cout << "[Parallel] iter " << (iter + 1) << ", residual = " << res
                  << " boundary_rhs_acc=" << /* per-dir last value */ 0.0 << std::endl;
        if (res < tol_) return res;
        prev_Tv = macro.Tv();
    }
    return macro.Residual(prev_Tv);
}

}  // namespace pbte

#endif  // MFEM_USE_MPI



