#pragma once

#include "AngularQuadrature.hpp"
#include "AngularSweepOrder.hpp"
#include "ElementIntegrator.hpp"
#include "MacroscopicQuantities.hpp"
#include "PhononProperties.hpp"

#include "mfem.hpp"

#include <map>
#include <unordered_map>
#include <optional>
#include <vector>

namespace pbte
{
enum class CachePolicy
{
    FullLU,     // precompute and store LU for every (dir,spec,cell)
    OnTheFly    // assemble A and factor per use (lower memory, slower)
};

/// Serial DG solver for non-gray SMRT PBTE (MFEM-based).
class PBTESolver
{
public:
    PBTESolver(const mfem::Mesh &mesh,
               const mfem::FiniteElementSpace &fes,
               const AngleQuadrature &angle_quad,
               const AngularSweepOrder &sweep_order,
               const std::vector<ElementIntegralData> &elem_data,
               const PhononProperties &props,
               const std::map<int, double> &isothermal_bcs,
               CachePolicy policy = CachePolicy::FullLU,
               double tol = 1e-8,
               int max_iter = 200);

    /// Solve in-place updating coeff[dir][branch][spec] (ndof x ne matrices).
    /// Returns achieved residual.
    double Solve(std::vector<std::vector<std::vector<mfem::DenseMatrix>>> &coeff,
                 MacroscopicQuantities &macro);

    /// Allocate and zero-initialize coefficient blocks sized to the problem.
    /// Layout: coeff[dir][branch][spec] -> (ndof x ne) matrices.
    std::vector<std::vector<std::vector<mfem::DenseMatrix>>> CreateInitialCoefficients() const;

private:
    struct FaceInfo
    {
        int face_id = -1;
        bool is_boundary = false;
        int boundary_attr = 0;
        mfem::Vector normal;             // outward for this element
        mfem::DenseMatrix coupling;      // element vs neighbor (if neighbor)
        mfem::DenseMatrix face_mass;     // M_face for outflow term
        mfem::Vector face_integral;      // ∫_F p_i (for thermal BC)
        int neighbor_elem = -1;          // neighbor element id (serial)
    };

    struct CellData
    {
        mfem::DenseMatrix mass;
        mfem::DenseMatrix mass_t;
        std::vector<mfem::DenseMatrix> stiff; // size = dim
        std::vector<FaceInfo> faces;
    };

    struct LUBlock
    {
        struct LUSolver
        {
            mfem::DenseMatrix A;
            std::unique_ptr<mfem::DenseMatrixInverse> inv;
            void Factor(const mfem::DenseMatrix &src);
            void Solve(const mfem::Vector &b, mfem::Vector &x) const;
        };
        std::vector<mfem::DenseMatrix> A;        // assembled A per cell
        std::vector<LUSolver> lu;                // LU per cell (cached or on-demand)
    };

    // Helpers
    void BuildCellData();
    mfem::Vector FaceNormal(int face_id, int elem_id) const;
    mfem::DenseMatrix AssembleA(int dir_idx, int branch, int spec, int elem) const;
    void EnsureLU(int dir_idx, int branch, int spec, int elem);

    // Members
    const mfem::Mesh &mesh_;
    const mfem::FiniteElementSpace &fes_;
    const AngleQuadrature &quad_;
    const AngularSweepOrder &sweep_;
    const std::vector<ElementIntegralData> &elem_data_;
    const PhononProperties &props_;
    std::map<int, double> iso_bc_;

    CachePolicy policy_;
    double tol_;
    int max_iter_;

    int dim_ = 0;
    int ndof_ = 0;
    int ne_ = 0;
    int nbranch_ = 0;
    int nspec_ = 0;
    int ndir_ = 0;
    double dt_inv_ = 0.0;
    double omega_ = 0.0;

    std::vector<CellData> cells_;

    // Optional LU cache: [dir][branch][spec] -> per-cell block
    std::vector<std::vector<std::vector<LUBlock>>> lu_cache_;
};

#ifdef MFEM_USE_MPI
/// Parallel solver skeleton (not implemented yet).
class PBTESolverPar
{
public:
    PBTESolverPar(const mfem::ParMesh &pmesh,
                  const mfem::ParFiniteElementSpace &pfes,
                  const AngleQuadrature &angle_quad,
                  const AngularSweepOrder &sweep_order,
                  const std::vector<ElementIntegralData> &elem_data,
                  const PhononProperties &props,
                  const std::map<int, double> &isothermal_bcs,
                  CachePolicy policy = CachePolicy::FullLU,
                  double tol = 1e-8,
                  int max_iter = 200);

    double Solve(std::vector<std::vector<std::vector<mfem::DenseMatrix>>> &coeff,
                 MacroscopicQuantities &macro);

private:
    struct FaceInfo
    {
        int face_id = -1;
        bool is_boundary = false;
        int boundary_attr = 0;
        mfem::Vector normal;
        mfem::DenseMatrix coupling;
        mfem::DenseMatrix face_mass;
        mfem::Vector face_integral;
        int neighbor_elem = -1;      // local interior neighbor
        bool is_shared = false;
        mfem::Array<int> face_nbr_vdofs;  // for shared face: vdofs into ParGridFunction (includes negative entries)
    };

    struct CellData
    {
        mfem::DenseMatrix mass;
        mfem::DenseMatrix mass_t;
        std::vector<mfem::DenseMatrix> stiff;
        std::vector<FaceInfo> faces;
    };

    struct LUBlock
    {
        struct LUSolver
        {
            mfem::DenseMatrix A;
            std::unique_ptr<mfem::DenseMatrixInverse> inv;
            void Factor(const mfem::DenseMatrix &src);
            void Solve(const mfem::Vector &b, mfem::Vector &x) const;
        };
        std::vector<mfem::DenseMatrix> A;
        std::vector<LUSolver> lu;
    };

    void BuildCellData();
    mfem::Vector FaceNormal(int face_id, int elem_id) const;
    mfem::DenseMatrix AssembleA(int dir_idx, int branch, int spec, int elem) const;
    void EnsureLU(int dir_idx, int branch, int spec, int elem);

    const mfem::ParMesh &pmesh_;
    const mfem::ParFiniteElementSpace &pfes_;
    const AngleQuadrature &quad_;
    const AngularSweepOrder &sweep_;
    const std::vector<ElementIntegralData> &elem_data_;
    const PhononProperties &props_;
    std::map<int, double> iso_bc_;

    CachePolicy policy_;
    double tol_;
    int max_iter_;

    int dim_ = 0;
    int ndof_ = 0;
    int ne_ = 0;          // local elements
    int nbranch_ = 0;
    int nspec_ = 0;
    int ndir_ = 0;
    double dt_inv_ = 0.0;
    double omega_ = 0.0;

    std::vector<CellData> cells_;
    std::unordered_map<int, int> face_to_shared_idx_; // face_id -> shared face idx

    std::vector<std::vector<std::vector<LUBlock>>> lu_cache_;
};
#endif

}  // namespace pbte


