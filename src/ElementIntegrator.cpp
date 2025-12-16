// Element-wise DG integrals (volume + face) for scalar L2 elements.
#include "ElementIntegrator.hpp"

#include "mfem.hpp"

namespace pbte
{
DGElementIntegrator::DGElementIntegrator(const mfem::FiniteElementSpace &fes,
                                         int quadrature_order)
    : fes_(fes),
      mesh_(*fes.GetMesh()),
      quadrature_order_(quadrature_order),
      int_rules_(0, mfem::Quadrature1D::GaussLegendre)
{
#ifdef MFEM_USE_MPI
    pfes_ = dynamic_cast<const mfem::ParFiniteElementSpace *>(&fes_);
    pmesh_ = (pfes_ != nullptr)
                 ? pfes_->GetParMesh()
                 : dynamic_cast<const mfem::ParMesh *>(&mesh_);
    is_parallel_ = (pfes_ != nullptr && pmesh_ != nullptr);
    if (pfes_)
    {
        // Ensure face-neighbor data is available.
        const_cast<mfem::ParFiniteElementSpace *>(pfes_)->ExchangeFaceNbrData();
    }
#else
    is_parallel_ = false;
#endif

    BuildFaceAttributes();
}

int DGElementIntegrator::EffectiveOrder(const mfem::FiniteElement &fe) const
{
    if (quadrature_order_ > 0)
    {
        return quadrature_order_;
    }
    // Default: exact for polynomial products up to degree 2*p.
    return 2 * fe.GetOrder() + 1;
}

ElementIntegralData DGElementIntegrator::AssembleElement(int elem_id)
{
    ElementIntegralData data;

    const mfem::FiniteElement *fe = fes_.GetFE(elem_id);
    mfem::ElementTransformation *tr = mesh_.GetElementTransformation(elem_id);
    const int ndof = fe->GetDof();
    const int dim = mesh_.Dimension();

    data.basis_integrals.SetSize(ndof);
    data.basis_integrals = 0.0;

    data.mass_matrix.SetSize(ndof);
    data.mass_matrix = 0.0;

    data.stiffness_matrices.resize(dim);
    for (auto &mat : data.stiffness_matrices)
    {
        mat.SetSize(ndof);
        mat = 0.0;
    }

    const mfem::IntegrationRule &ir =
        int_rules_.Get(fe->GetGeomType(), EffectiveOrder(*fe));

    mfem::Vector shape(ndof);
    mfem::DenseMatrix dshape(ndof, dim);

    for (int q = 0; q < ir.GetNPoints(); ++q)
    {
        const mfem::IntegrationPoint &ip = ir.IntPoint(q);
        tr->SetIntPoint(&ip);
        const double w = ip.weight * tr->Weight();

        fe->CalcShape(ip, shape);
        fe->CalcPhysDShape(*tr, dshape);

        for (int i = 0; i < ndof; ++i)
        {
            const double si = shape(i);
            data.basis_integrals(i) += w * si;

            for (int j = 0; j < ndof; ++j)
            {
                const double sj = shape(j);
                data.mass_matrix(i, j) += w * si * sj;

                for (int d = 0; d < dim; ++d)
                {
                    data.stiffness_matrices[d](i, j) += w * dshape(i, d) * sj;
                }
            }
        }
    }

    return data;
}

void DGElementIntegrator::AssembleFaceContributions(
    std::vector<ElementIntegralData> &edata)
{
    const int nfaces = mesh_.GetNumFaces();
    for (int face_id = 0; face_id < nfaces; ++face_id)
    {
        mfem::FaceElementTransformations *ftr =
            mesh_.GetFaceElementTransformations(face_id);
        if (!ftr)
        {
            continue;
        }

        const int elem_ids[2] = {ftr->Elem1No, ftr->Elem2No};
        mfem::ElementTransformation *el_trs[2] = {ftr->Elem1, ftr->Elem2};
        const int face_attr =
            (face_id >= 0 && face_id < static_cast<int>(face_attributes_.size()))
                ? face_attributes_[face_id]
                : 0;

        for (int side = 0; side < 2; ++side)
        {
            const int elem_id = elem_ids[side];
            if (elem_id < 0)
            {
                // On ParMesh, shared faces appear with Elem2No < 0; skip here
                // and process in shared-face loop.
                if (is_parallel_)
                {
                    continue;
                }
                // boundary face for the other side (serial)
                continue;
            }

            const mfem::FiniteElement *fe = fes_.GetFE(elem_id);
            const int ndof = fe->GetDof();

            mfem::DenseMatrix face_mass;
            mfem::Vector face_int;
            face_mass.SetSize(ndof);
            face_mass = 0.0;
            face_int.SetSize(ndof);
            face_int = 0.0;

            mfem::DenseMatrix coupling;
            const int neigh_id = elem_ids[1 - side];
            const bool has_neighbor = (neigh_id >= 0);
            const mfem::FiniteElement *fe_neigh = has_neighbor ? fes_.GetFE(neigh_id)
                                                               : nullptr;
            int ndof_neigh = has_neighbor ? fe_neigh->GetDof() : 0;
            if (has_neighbor)
            {
                coupling.SetSize(ndof, ndof_neigh);
                coupling = 0.0;
            }

            const mfem::IntegrationRule &ir = int_rules_.Get(
                ftr->GetGeometryType(), EffectiveOrder(*fe));

            mfem::Vector shape(ndof);
            mfem::Vector shape_neigh;
            if (has_neighbor)
            {
                shape_neigh.SetSize(ndof_neigh);
            }

            for (int q = 0; q < ir.GetNPoints(); ++q)
            {
                const mfem::IntegrationPoint &ip_face = ir.IntPoint(q);
                ftr->Face->SetIntPoint(&ip_face);

                const mfem::IntegrationPoint &ip_el =
                    (side == 0) ? ftr->GetElement1IntPoint()
                                : ftr->GetElement2IntPoint();
                el_trs[side]->SetIntPoint(&ip_el);

                fe->CalcShape(ip_el, shape);
                if (has_neighbor)
                {
                    const mfem::IntegrationPoint &ip_el_neigh =
                        (side == 0) ? ftr->GetElement2IntPoint()
                                    : ftr->GetElement1IntPoint();
                    el_trs[1 - side]->SetIntPoint(&ip_el_neigh);
                    fe_neigh->CalcShape(ip_el_neigh, shape_neigh);
                }

                const double w = ip_face.weight * ftr->Face->Weight();

                for (int i = 0; i < ndof; ++i)
                {
                    const double si = shape(i);
                    face_int(i) += w * si;

                    for (int j = 0; j < ndof; ++j)
                    {
                        face_mass(i, j) += w * si * shape(j);
                    }

                    if (has_neighbor)
                    {
                        for (int j = 0; j < ndof_neigh; ++j)
                        {
                            coupling(i, j) += w * si * shape_neigh(j);
                        }
                    }
                }
            }

            ElementIntegralData &edata_el = edata[elem_id];
            edata_el.face_mass_matrices.emplace_back();
            edata_el.face_mass_matrices.back().Swap(face_mass);
            edata_el.face_integrals.emplace_back();

            ElementIntegralData::FaceCoupling fc;
            fc.face_id = face_id;
            fc.boundary_attr = face_attr;
            fc.neighbor_elem = has_neighbor ? neigh_id : -1;
            fc.is_shared = false;
            if (has_neighbor)
            {
                fc.coupling.Swap(coupling);
            }
            else
            {
                fc.isothermal_rhs.SetSize(ndof);
                fc.isothermal_rhs = face_int;  // p_i * 1
            }
            edata_el.face_couplings.push_back(std::move(fc));

            // After recording boundary RHS and couplings, move face data into
            // storage vectors.
            edata_el.face_integrals.back().Swap(face_int);
        }
    }
}

void DGElementIntegrator::BuildFaceAttributes()
{
    const int nfaces = mesh_.GetNumFaces();
    face_attributes_.assign(nfaces, 0);
    // Map boundary elements to faces and record their attributes.
    const int nbe = mesh_.GetNBE();
    for (int be = 0; be < nbe; ++be)
    {
        const int face = mesh_.GetBdrElementFaceIndex(be);
        if (face >= 0 && face < nfaces)
        {
            face_attributes_[face] = mesh_.GetBdrAttribute(be);
        }
    }
}

void DGElementIntegrator::AssembleSharedFaceContributions(
    std::vector<ElementIntegralData> &edata)
{
#ifndef MFEM_USE_MPI
    (void)edata;
    return;
#else
    if (!is_parallel_ || !pfes_ || !pmesh_)
    {
        return;
    }

    const int nshf = pmesh_->GetNSharedFaces();
    for (int sf = 0; sf < nshf; ++sf)
    {
        mfem::FaceElementTransformations *ftr =
            pmesh_->GetSharedFaceTransformations(sf, /*fill2=*/true);
        if (!ftr)
        {
            continue;
        }
        const int face_id = pmesh_->GetSharedFace(sf);
        const int elem_id = ftr->Elem1No;  // local element
        if (elem_id < 0)
        {
            continue;
        }

        const mfem::FiniteElement *fe = fes_.GetFE(elem_id);
        const int ndof = fe->GetDof();

        // Neighbor data (face neighbor element).
        const mfem::FiniteElement *fe_neigh = pfes_->GetFaceNbrFE(sf);
        if (!fe_neigh)
        {
            continue;
        }
        const int ndof_neigh = fe_neigh->GetDof();

        mfem::DenseMatrix coupling(ndof, ndof_neigh);
        coupling = 0.0;

        const mfem::IntegrationRule &ir = int_rules_.Get(
            ftr->GetGeometryType(), EffectiveOrder(*fe));

        mfem::Vector shape(ndof);
        mfem::Vector shape_neigh(ndof_neigh);

        for (int q = 0; q < ir.GetNPoints(); ++q)
        {
            const mfem::IntegrationPoint &ip_face = ir.IntPoint(q);
            ftr->Face->SetIntPoint(&ip_face);

            const mfem::IntegrationPoint &ip_el =
                ftr->GetElement1IntPoint();  // local
            ftr->Elem1->SetIntPoint(&ip_el);
            fe->CalcShape(ip_el, shape);

            const mfem::IntegrationPoint &ip_el_neigh =
                ftr->GetElement2IntPoint();  // neighbor
            ftr->Elem2->SetIntPoint(&ip_el_neigh);
            fe_neigh->CalcShape(ip_el_neigh, shape_neigh);

            const double w = ip_face.weight * ftr->Face->Weight();
            for (int i = 0; i < ndof; ++i)
            {
                const double si = shape(i);
                for (int j = 0; j < ndof_neigh; ++j)
                {
                    coupling(i, j) += w * si * shape_neigh(j);
                }
            }
        }

        ElementIntegralData &edata_el = edata[elem_id];
        ElementIntegralData::FaceCoupling fc;
        fc.face_id = face_id;
        fc.boundary_attr = 0;
        fc.neighbor_elem = -2;  // shared neighbor
        fc.is_shared = true;
        fc.coupling.Swap(coupling);
        edata_el.face_couplings.push_back(std::move(fc));
    }
#endif
}

std::vector<ElementIntegralData> DGElementIntegrator::AssembleAll()
{
    std::vector<ElementIntegralData> results;
    results.reserve(mesh_.GetNE());

    for (int e = 0; e < mesh_.GetNE(); ++e)
    {
        results.emplace_back(AssembleElement(e));
    }

    AssembleFaceContributions(results);
    return results;
}
}  // namespace pbte

