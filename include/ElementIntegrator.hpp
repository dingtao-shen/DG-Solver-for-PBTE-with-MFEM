// DG element/face integrals for L2 finite elements using MFEM.
#pragma once

#include "mfem.hpp"

#include <vector>

namespace pbte
{
/// Stores all local integral objects for one element.
struct ElementIntegralData
{
    /// \int_K p_i
    mfem::Vector basis_integrals;
    /// \int_K p_i p_j
    mfem::DenseMatrix mass_matrix;
    /// \int_K \partial_{x_d} p_i \, p_j  (one matrix per spatial dimension)
    std::vector<mfem::DenseMatrix> stiffness_matrices;
    /// \int_{F} p_i p_j for each face of the element (same element both sides).
    std::vector<mfem::DenseMatrix> face_mass_matrices;
    /// \int_{F} p_i for each face of the element (basis vs constant 1).
    std::vector<mfem::Vector> face_integrals;

    /// Coupling with neighbor elements across faces, or boundary (neighbor=-1).
    struct FaceCoupling
    {
        int face_id = -1;
        int neighbor_elem = -1;   // -1 means boundary; -2 marks shared neighbor
        int boundary_attr = 0;     // only meaningful if neighbor_elem == -1
        bool is_shared = false;    // true if neighbor is on another rank
        mfem::DenseMatrix coupling;  // \int_F p_i(elem) * p_j(neigh)
        mfem::Vector isothermal_rhs; // \int_F p_i(elem) * 1  (for boundary)
    };
    std::vector<FaceCoupling> face_couplings;
};

/// Element-wise DG integrator that assembles scalar (L2) basis integrals,
/// mass matrices, stiffness matrices, and face integrals.
class DGElementIntegrator
{
public:
    /// quadrature_order <= 0 uses a safe default: 2*p + 1 where p is FE order.
    DGElementIntegrator(const mfem::FiniteElementSpace &fes,
                        int quadrature_order = -1);

    /// Assemble all requested quantities for every element (in serial).
    std::vector<ElementIntegralData> AssembleAll();

    const mfem::FiniteElementSpace &FESpace() const { return fes_; }

private:
    int EffectiveOrder(const mfem::FiniteElement &fe) const;
    ElementIntegralData AssembleElement(int elem_id);
    void AssembleFaceContributions(std::vector<ElementIntegralData> &data);
    void AssembleSharedFaceContributions(std::vector<ElementIntegralData> &data);
    void BuildFaceAttributes();

    const mfem::FiniteElementSpace &fes_;
    mfem::Mesh &mesh_;  // non-const: element/face transforms are non-const in MFEM
#ifdef MFEM_USE_MPI
    const mfem::ParFiniteElementSpace *pfes_ = nullptr;
    mfem::ParMesh *pmesh_ = nullptr;
#else
    const void *pfes_ = nullptr;
    const void *pmesh_ = nullptr;
#endif
    bool is_parallel_ = false;
    int quadrature_order_;
    mfem::IntegrationRules int_rules_;
    std::vector<int> face_attributes_;
};
}  // namespace pbte

