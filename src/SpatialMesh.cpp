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

void SpatialMesh::UniformRefine(int levels)
{
    if (!mesh_ || levels <= 0)
    {
        return;
    }
#ifdef MFEM_USE_MPI
    if (last_parallel_ && pmesh_)
    {
        for (int i = 0; i < levels; ++i)
        {
            pmesh_->UniformRefinement();
        }
        return;
    }
#endif
    for (int i = 0; i < levels; ++i)
    {
        mesh_->UniformRefinement();
    }
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

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line))
    {
        const auto hash_pos = line.find('#');
        if (hash_pos != std::string::npos)
        {
            line = line.substr(0, hash_pos);
        }
        lines.push_back(line);
    }

    std::string path_val;
    bool in_mesh_block = false;
    for (const auto &raw : lines)
    {
        std::string l = trim(raw);
        if (l.empty())
        {
            continue;
        }
        if (l.rfind("mesh:", 0) == 0)
        {
            in_mesh_block = true;
            continue;
        }
        if (in_mesh_block && l.rfind("path:", 0) == 0)
        {
            path_val = trim(l.substr(std::string("path:").size()));
            break;
        }
        if (!in_mesh_block && l.rfind("mesh_path:", 0) == 0)
        {
            path_val = trim(l.substr(std::string("mesh_path:").size()));
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

    // Parse optional boundary temperatures
    isothermal_bc_.clear();
    bool in_bc_block = false;
    int current_attr = -1;
    auto flush_entry = [&]() { current_attr = -1; };
    for (const auto &raw : lines)
    {
        std::string l = trim(raw);
        if (l.empty())
        {
            continue;
        }
        if (l.rfind("boundary_conditions:", 0) == 0)
        {
            in_bc_block = true;
            continue;
        }
        if (!in_bc_block)
        {
            continue;
        }
        if (l.rfind("-", 0) == 0)
        {
            flush_entry();
            // parse inline after dash if present
            const std::string rest = trim(l.substr(1));
            if (rest.rfind("attr:", 0) == 0)
            {
                const std::string val = trim(rest.substr(std::string("attr:").size()));
                current_attr = std::stoi(val);
            }
            else if (rest.rfind("temperature:", 0) == 0 && current_attr >= 0)
            {
                const std::string val = trim(rest.substr(std::string("temperature:").size()));
                try
                {
                    const double t = std::stod(val);
                    isothermal_bc_[current_attr] = t;
                }
                catch (const std::exception &)
                {
                    std::cerr << "Warning: failed to parse temperature value: " << l
                              << std::endl;
                }
            }
            continue;
        }
        if (l.rfind("attr:", 0) == 0)
        {
            const std::string val = trim(l.substr(std::string("attr:").size()));
            current_attr = std::stoi(val);
            continue;
        }
        if (l.rfind("temperature:", 0) == 0 && current_attr >= 0)
        {
            const std::string val =
                trim(l.substr(std::string("temperature:").size()));
            try
            {
                const double t = std::stod(val);
                isothermal_bc_[current_attr] = t;
            }
            catch (const std::exception &)
            {
                std::cerr << "Warning: failed to parse temperature value: " << l
                          << std::endl;
            }
            flush_entry();
        }
    }
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

    // nodal basis function based on Gauss-Lobatto quadrature
    fec_ = std::make_unique<mfem::L2_FECollection>(order, mesh_->Dimension(),mfem::BasisType::ClosedUniform);
    // fec_ = std::make_unique<mfem::L2_FECollection>(order, mesh_->Dimension());
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

    // Partition serial mesh into ParMesh (no user partitioning provided).
    pmesh_ = std::make_unique<mfem::ParMesh>(comm, *mesh_,
                                             /*partitioning=*/nullptr,
                                             /*part_method=*/1);
    pmesh_->ExchangeFaceNbrData();

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

mfem::FiniteElementSpace *SpatialMesh::ActiveFESpace()
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
