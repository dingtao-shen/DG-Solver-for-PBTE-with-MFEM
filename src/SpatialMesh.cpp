// SpatialMesh.cpp
#include "SpatialMesh.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace fs = std::filesystem;

namespace pbte
{
namespace
{
// Default resolutions for built-in meshes.
constexpr int kDefaultNx2D = 8;
constexpr int kDefaultNy2D = 8;
constexpr int kDefaultNx3D = 4;
constexpr int kDefaultNy3D = 4;
constexpr int kDefaultNz3D = 4;
}  // namespace

void SpatialMesh::LoadMesh(const std::string &path_or_builtin)
{
    mesh_source_ = path_or_builtin;
    if (path_or_builtin.empty())
    {
        throw std::invalid_argument("Mesh path or builtin name cannot be empty.");
    }

    if (fs::exists(path_or_builtin))
    {
        mesh_ = std::make_unique<mfem::Mesh>(path_or_builtin.c_str(), 1, 1);
        return;
    }

    LoadBuiltin(path_or_builtin);
}

void SpatialMesh::LoadMeshFromConfig(const std::string &config_path)
{
    std::ifstream in(config_path);
    if (!in)
    {
        throw std::runtime_error("Failed to open config file: " + config_path);
    }

    auto trim = [](const std::string &s) -> std::string {
        const auto first = s.find_first_not_of(" \t\r\n");
        if (first == std::string::npos)
        {
            return "";
        }
        const auto last = s.find_last_not_of(" \t\r\n");
        return s.substr(first, last - first + 1);
    };

    bool in_mesh_block = false;
    std::string path_val;
    std::string line;
    while (std::getline(in, line))
    {
        // strip comments
        const auto hash_pos = line.find('#');
        if (hash_pos != std::string::npos)
        {
            line = line.substr(0, hash_pos);
        }
        line = trim(line);
        if (line.empty())
        {
            continue;
        }

        if (line.rfind("mesh:", 0) == 0)
        {
            in_mesh_block = true;
            continue;
        }

        if (in_mesh_block && line.rfind("path:", 0) == 0)
        {
            path_val = trim(line.substr(std::string("path:").size()));
            break;
        }

        if (!in_mesh_block && line.rfind("mesh_path:", 0) == 0)
        {
            path_val = trim(line.substr(std::string("mesh_path:").size()));
            break;
        }
    }

    if (path_val.empty())
    {
        throw std::runtime_error("No mesh path found in config: " + config_path);
    }

    // Remove optional surrounding quotes
    if ((path_val.front() == '\'' && path_val.back() == '\'') ||
        (path_val.front() == '"' && path_val.back() == '"'))
    {
        path_val = path_val.substr(1, path_val.size() - 2);
    }

    LoadMesh(path_val);
}

void SpatialMesh::BuildDGSpace(int order, mfem::Ordering::Type ordering,
                               const std::string &log_path)
{
    if (!mesh_)
    {
        throw std::runtime_error("Mesh must be loaded before building FE space.");
    }
    if (order < 0)
    {
        throw std::invalid_argument("Polynomial order must be non-negative.");
    }

    last_parallel_ = false;
    last_order_ = order;

    fec_ = std::make_unique<mfem::L2_FECollection>(order, mesh_->Dimension());
    fes_ = std::make_unique<mfem::FiniteElementSpace>(mesh_.get(), fec_.get(), 1,
                                                      ordering);

    LogSummary(log_path);
}

#ifdef MFEM_USE_MPI
void SpatialMesh::BuildDGSpaceParallel(MPI_Comm comm, int order,
                                       mfem::Ordering::Type ordering,
                                       const std::string &log_path)
{
    if (!mesh_)
    {
        throw std::runtime_error("Mesh must be loaded before building FE space.");
    }
    if (order < 0)
    {
        throw std::invalid_argument("Polynomial order must be non-negative.");
    }

    last_parallel_ = true;
    last_comm_ = comm;
    MPI_Comm_size(comm, &mpi_size_);
    MPI_Comm_rank(comm, &mpi_rank_);
    last_order_ = order;

    pmesh_ = std::make_unique<mfem::ParMesh>(comm, *mesh_, /*refine=*/0,
                                             /*partition=*/0,
                                             /*fix_orientation=*/true);

    fec_ = std::make_unique<mfem::L2_FECollection>(order, pmesh_->Dimension());
    pfes_ = std::make_unique<mfem::ParFiniteElementSpace>(pmesh_.get(),
                                                          fec_.get(), 1,
                                                          ordering);

    LogSummary(log_path);
}
#endif

void SpatialMesh::LoadBuiltin(const std::string &name)
{
    if (name == "unit-square" || name == "unit-square-tri")
    {
        mesh_ = std::make_unique<mfem::Mesh>(mfem::Mesh::MakeCartesian2D(
            kDefaultNx2D, kDefaultNy2D, mfem::Element::TRIANGLE,
            /*generate_edges=*/true, 1.0, 1.0));
        return;
    }

    if (name == "unit-square-quad")
    {
        mesh_ = std::make_unique<mfem::Mesh>(mfem::Mesh::MakeCartesian2D(
            kDefaultNx2D, kDefaultNy2D, mfem::Element::QUADRILATERAL,
            /*generate_edges=*/true, 1.0, 1.0));
        return;
    }

    if (name == "unit-cube" || name == "unit-cube-tet")
    {
        mesh_ = std::make_unique<mfem::Mesh>(mfem::Mesh::MakeCartesian3D(
            kDefaultNx3D, kDefaultNy3D, kDefaultNz3D, mfem::Element::TETRAHEDRON,
            1.0, 1.0, 1.0));
        return;
    }

    if (name == "unit-cube-hex")
    {
        mesh_ = std::make_unique<mfem::Mesh>(mfem::Mesh::MakeCartesian3D(
            kDefaultNx3D, kDefaultNy3D, kDefaultNz3D, mfem::Element::HEXAHEDRON,
            1.0, 1.0, 1.0));
        return;
    }

    throw std::invalid_argument("Unrecognized built-in mesh name: " + name);
}

std::string SpatialMesh::MakeLogPath(const std::string &log_path) const
{
    if (!log_path.empty())
    {
        return log_path;
    }

    // Auto-generate: output/log/mesh_<source>_p<order>_dim<dim>.txt
    std::string base;
    try
    {
        fs::path p(mesh_source_);
        base = p.has_stem() ? p.stem().string() : mesh_source_;
    }
    catch (...)
    {
        base = mesh_source_;
    }
    if (base.empty())
    {
        base = "mesh";
    }
    for (char &c : base)
    {
        if (c == '/' || c == '\\' || c == ' ')
        {
            c = '_';
        }
    }

    std::ostringstream oss;
    oss << "output/log/mesh_" << base << "_p" << last_order_ << "_dim"
        << ActiveMesh()->Dimension();
    if (last_parallel_)
    {
        oss << "_par";
#ifdef MFEM_USE_MPI
        oss << "_r" << mpi_rank_;
#endif
    }
    oss << ".txt";
    return oss.str();
}

std::string SpatialMesh::MakeSummary() const
{
    std::ostringstream report;
    const mfem::Mesh *m = ActiveMesh();
    const mfem::FiniteElementSpace *f = ActiveFESpace();

    report << "Mesh and DG space summary\n";
    report << "  mesh source          : " << mesh_source_ << "\n";
    report << "  dimension            : " << m->Dimension() << "\n";
    report << "  element count        : " << m->GetNE() << "\n";
    report << "  boundary elem count  : " << m->GetNBE() << "\n";
    report << "  vertex count         : " << m->GetNV() << "\n";
    const int ne = m->GetNE();
    std::string geom_name =
        (ne > 0) ? mfem::Geometry::Name[m->GetElementBaseGeometry(0)]
                 : "unknown";
    report << "  element geometry     : " << geom_name << "\n";
    report << "  DG polynomial order  : " << last_order_ << "\n";
    report << "  FE space ndofs       : " << f->GetNDofs() << "\n";
    report << "  FE space vdim        : " << f->GetVDim() << "\n";
    report << "  ordering             : "
           << (f->GetOrdering() == mfem::Ordering::byNODES ? "byNODES"
                                                           : "byVDIM")
           << "\n";
#ifdef MFEM_USE_MPI
    if (last_parallel_ && pmesh_)
    {
        report << "  mpi size/rank        : " << mpi_size_ << "/" << mpi_rank_
               << "\n";
        report << "  global elements      : " << pmesh_->GetGlobalNE() << "\n";
        report << "  global vertices      : " << pmesh_->GetGlobalNV() << "\n";
        report << "  global true dofs     : " << pfes_->GlobalTrueVSize() << "\n";
    }
#endif
    return report.str();
}

void SpatialMesh::LogSummary(const std::string &log_path) const
{
    const std::string summary = MakeSummary();
    bool should_print = true;
#ifdef MFEM_USE_MPI
    if (last_parallel_)
    {
        should_print = (mpi_rank_ == 0);
    }
#endif
    if (should_print)
    {
        std::cout << summary << std::endl;
    }

    try
    {
        const std::string resolved = MakeLogPath(log_path);
        if (!resolved.empty())
        {
            fs::path path(resolved);
            fs::create_directories(path.parent_path());
            std::ofstream ofs(path);
            if (ofs)
            {
                ofs << summary;
                std::cout << "Log written to: " << path << std::endl;
            }
            else
            {
                std::cerr << "Failed to open log file for writing: " << path
                          << std::endl;
            }
        }
    }
    catch (const std::exception &ex)
    {
        std::cerr << "Failed to write log: " << ex.what() << std::endl;
    }
}

const mfem::Mesh *SpatialMesh::ActiveMesh() const
{
#ifdef MFEM_USE_MPI
    if (last_parallel_ && pmesh_)
    {
        return pmesh_.get();
    }
#endif
    return mesh_.get();
}

const mfem::FiniteElementSpace *SpatialMesh::ActiveFESpace() const
{
#ifdef MFEM_USE_MPI
    if (last_parallel_ && pfes_)
    {
        return pfes_.get();
    }
#endif
    return fes_.get();
}
}  // namespace pbte
