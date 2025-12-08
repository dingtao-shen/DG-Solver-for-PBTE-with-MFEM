// SpatialMesh.hpp
// Lightweight helper to load an mfem::Mesh (from file or built-in generators)
// and to build a discontinuous (DG/L2) finite element space of configurable order.
#pragma once

#include <memory>
#include <ostream>
#include <string>

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif

#include "mfem.hpp"

namespace pbte
{
/// Encapsulates mesh loading and DG finite element space construction.
class SpatialMesh
{
public:
    SpatialMesh() = default;

    /// Load a mesh either from a file path or from a small set of built-in
    /// generator names. Recognized built-ins (case sensitive):
    ///   - "unit-square":      Cartesian 2D triangular mesh (default 8x8).
    ///   - "unit-square-quad": Cartesian 2D quad mesh (default 8x8).
    ///   - "unit-square-tri":  Cartesian 2D triangular mesh (default 8x8).
    ///   - "unit-cube":        Cartesian 3D tetrahedral mesh (default 4x4x4).
    ///   - "unit-cube-hex":    Cartesian 3D hexahedral mesh (default 4x4x4).
    ///   - "unit-cube-tet":    Cartesian 3D tetrahedral mesh (default 4x4x4).
    /// If the string resolves to an existing file, the file is loaded.
    void LoadMesh(const std::string &path_or_builtin);

    /// Load mesh using a config YAML at config_path.
    /// Expected keys (any one):
    ///   mesh:
    ///     path: config/mesh/xxx.mesh
    ///   mesh_path: config/mesh/xxx.mesh
    /// Throws if missing/empty.
    void LoadMeshFromConfig(const std::string &config_path);

    /// Build a discontinuous finite element space of the given polynomial order
    /// on the already-loaded mesh. Uses mfem::L2_FECollection which is
    /// appropriate for DG formulations.
    // Build DG space and emit a summary. If log_path is empty, a filename
    // embedding mesh source and order is auto-generated under output/log/.
    void BuildDGSpace(
        int order,
        mfem::Ordering::Type ordering = mfem::Ordering::byNODES,
        const std::string &log_path = "");

#ifdef MFEM_USE_MPI
    /// Build a parallel DG space (ParMesh + ParFiniteElementSpace).
    /// Requires MFEM built with MPI. Uses the already-loaded serial mesh as
    /// input for partitioning.
    void BuildDGSpaceParallel(
        MPI_Comm comm,
        int order,
        mfem::Ordering::Type ordering = mfem::Ordering::byNODES,
        const std::string &log_path = "");

    /// Accessors (non-owning, only valid after parallel build).
    mfem::ParMesh *ParMeshPtr() { return pmesh_.get(); }
    const mfem::ParMesh *ParMeshPtr() const { return pmesh_.get(); }
    mfem::ParFiniteElementSpace *ParFESpacePtr() { return pfes_.get(); }
    const mfem::ParFiniteElementSpace *ParFESpacePtr() const { return pfes_.get(); }
#endif

    /// Accessors (non-owning).
    mfem::Mesh &Mesh() { return *mesh_; }
    const mfem::Mesh &Mesh() const { return *mesh_; }
    mfem::FiniteElementSpace &FESpace() { return *fes_; }
    const mfem::FiniteElementSpace &FESpace() const { return *fes_; }
    int Dimension() const { return mesh_ ? mesh_->Dimension() : 0; }

private:
    void LoadBuiltin(const std::string &name);
    std::string MakeLogPath(const std::string &log_path) const;
    std::string MakeSummary() const;
    void LogSummary(const std::string &log_path) const;
    const mfem::Mesh *ActiveMesh() const;
    const mfem::FiniteElementSpace *ActiveFESpace() const;

    std::string mesh_source_;
    int last_order_ = -1;
    bool last_parallel_ = false;
#ifdef MFEM_USE_MPI
    MPI_Comm last_comm_ = MPI_COMM_NULL;
    int mpi_size_ = 1;
    int mpi_rank_ = 0;
#endif
    std::unique_ptr<mfem::Mesh> mesh_;
    std::unique_ptr<mfem::FiniteElementCollection> fec_;
    std::unique_ptr<mfem::FiniteElementSpace> fes_;
#ifdef MFEM_USE_MPI
    std::unique_ptr<mfem::ParMesh> pmesh_;
    std::unique_ptr<mfem::ParFiniteElementSpace> pfes_;
#endif
};
}  // namespace pbte