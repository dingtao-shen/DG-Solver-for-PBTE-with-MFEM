#include "AngularQuadrature.hpp"
#include "AngularSweepOrder.hpp"
#include "ElementIntegrator.hpp"
#include "SpatialMesh.hpp"

#include "mfem.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;

namespace
{
void WriteVector(std::ostream &os, const mfem::Vector &v,
                 const std::string &label)
{
    os << label << " [size=" << v.Size() << "]:";
    for (int i = 0; i < v.Size(); ++i)
    {
        os << " " << v[i];
    }
    os << "\n";
}

void WriteMatrix(std::ostream &os, const mfem::DenseMatrix &m,
                 const std::string &label)
{
    os << label << " [shape=" << m.Height() << "x" << m.Width() << "]\n";
    for (int i = 0; i < m.Height(); ++i)
    {
        os << "  ";
        for (int j = 0; j < m.Width(); ++j)
        {
            os << m(i, j);
            if (j + 1 < m.Width())
            {
                os << " ";
            }
        }
        os << "\n";
    }
}
}  // namespace

int main(int argc, char *argv[])
{
    std::string mesh_spec;  // optional override; if empty, read from config
    std::string config_path = "config/config.yaml";
    int order = 1;
    int refine_levels = 0;
    bool use_parallel = false;
    int angle_dim_cli = -1;
    int polar_pts_cli = -1;
    int azimuth_pts_cli = -1;
    std::string polar_scheme_cli;
    std::string azimuth_scheme_cli;

#ifdef MFEM_USE_MPI
    mfem::MPI_Session mpi(argc, argv);
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    const bool is_root = (mpi_rank == 0);
#else
    const bool is_root = true;
#endif

    mfem::OptionsParser args(argc, argv);
    args.AddOption(&mesh_spec, "-m", "--mesh",
                   "Mesh file to load or builtin name "
                   "(unit-square|unit-square-tri|unit-square-quad|"
                   "unit-cube|unit-cube-tet|unit-cube-hex). If omitted, "
                   "config/config.yaml is used.");
    args.AddOption(&config_path, "-c", "--config",
                   "Path to config YAML (defaults to config/config.yaml).");
    args.AddOption(&order, "-o", "--order",
                   "Polynomial order for DG/L2 finite element space.");
    args.AddOption(&refine_levels, "-r", "--refine",
                   "Uniform refinement levels (applied before building space).");
    args.AddOption(&use_parallel,
                   "-p", "--parallel",
                   "-np", "--no-parallel",
                   "Use parallel mesh/space (requires MFEM built with MPI).",
                   false);
    args.AddOption(&angle_dim_cli, "-ad", "--angle-dim",
                   "Angular quadrature dimension (2 or 3). Negative uses config.");
    args.AddOption(&polar_pts_cli, "-ap", "--polar-pts",
                   "Number of polar (theta) points. <=0 uses config.");
    args.AddOption(&azimuth_pts_cli, "-az", "--azimuth-pts",
                   "Number of azimuth (phi) points. <=0 uses config.");
    args.AddOption(&polar_scheme_cli, "-aps", "--polar-scheme",
                   "Polar scheme: uniform | gauss. Empty uses config.");
    args.AddOption(&azimuth_scheme_cli, "-aas", "--azimuth-scheme",
                   "Azimuth scheme: uniform | gauss. Empty uses config.");
    args.Parse();
    if (!args.Good())
    {
        if (is_root)
        {
            args.PrintUsage(std::cout);
        }
        return 1;
    }
    if (is_root)
    {
        args.PrintOptions(std::cout);
    }

#ifndef MFEM_USE_MPI
    if (use_parallel)
    {
        if (is_root)
        {
            std::cerr
                << "Error: -p/--parallel requested but MFEM_USE_MPI is not "
                   "defined (built without MPI). Rebuild with MPI-enabled MFEM "
                   "and rerun.\n";
        }
        return 1;
    }
#endif

    pbte::SpatialMesh spatial;
    try
    {
        if (!mesh_spec.empty())
        {
            spatial.LoadMesh(mesh_spec);
        }
        else
        {
            spatial.LoadMeshFromConfig(config_path);
        }
        spatial.UniformRefine(refine_levels);
#ifdef MFEM_USE_MPI
        if (use_parallel)
        {
            spatial.BuildDGSpaceParallel(MPI_COMM_WORLD, order);
        }
        else
#endif
        {
            spatial.BuildDGSpace(order);
        }
    }
    catch (const std::exception &ex)
    {
        std::cerr << "Error constructing mesh/FE space: " << ex.what() << "\n";
        return 1;
    }

    pbte::AngleDiscretizationOptions angle_opts;
    try
    {
        angle_opts = pbte::AngleQuadrature::LoadOptionsFromConfig(config_path);
    }
    catch (const std::exception &ex)
    {
        if (is_root)
        {
            std::cerr << "Warning: failed to read angular options from config: "
                      << ex.what()
                      << ". Using built-in defaults.\n";
        }
        angle_opts = pbte::AngleDiscretizationOptions{};
    }

    if (angle_dim_cli > 0)
    {
        angle_opts.dimension = angle_dim_cli;
    }
    if (polar_pts_cli > 0)
    {
        angle_opts.polar_points = polar_pts_cli;
    }
    if (azimuth_pts_cli > 0)
    {
        angle_opts.azimuth_points = azimuth_pts_cli;
    }
    if (!polar_scheme_cli.empty())
    {
        angle_opts.polar_scheme =
            pbte::AngleQuadrature::ParseSchemeName(polar_scheme_cli);
    }
    if (!azimuth_scheme_cli.empty())
    {
        angle_opts.azimuth_scheme =
            pbte::AngleQuadrature::ParseSchemeName(azimuth_scheme_cli);
    }

    pbte::AngleQuadrature angle_quad;
    try
    {
        angle_quad = pbte::AngleQuadrature::Build(angle_opts);
    }
    catch (const std::exception &ex)
    {
        std::cerr << "Error constructing angular quadrature: " << ex.what()
                  << "\n";
        return 1;
    }
    if (is_root)
    {
        std::cout << "Angular quadrature built: dim=" << angle_opts.dimension
                  << ", polar pts=" << angle_quad.PolarAngles().size()
                  << ", azimuth pts=" << angle_quad.AzimuthAngles().size()
                  << ", directions=" << angle_quad.Directions().size()
                  << ", total weight=" << angle_quad.TotalWeight() << std::endl;

        // Write angular quadrature to log.
        const std::string polar_scheme =
            pbte::AngleQuadrature::SchemeName(angle_opts.polar_scheme);
        const std::string azimuth_scheme =
            pbte::AngleQuadrature::SchemeName(angle_opts.azimuth_scheme);
        std::ostringstream ang_path;
        ang_path << "output/log/angles_dim" << angle_opts.dimension
                 << "_tp" << angle_opts.polar_points << "_" << polar_scheme
                 << "_ap" << angle_opts.azimuth_points << "_" << azimuth_scheme
                 << ".txt";
        try
        {
            angle_quad.WriteToFile(ang_path.str());
            std::cout << "Angular quadrature written to: " << ang_path.str()
                      << std::endl;
        }
        catch (const std::exception &log_ex)
        {
            std::cerr << "Failed to write angular log: " << log_ex.what()
                      << std::endl;
        }

        // Build sweep order (element ordering) for each direction and log it.
        try
        {
            const auto sweep = pbte::AngularSweepOrder::Build(spatial.Mesh(), angle_quad);
            std::ostringstream sw_path;
            sw_path << "output/log/sweep_dim" << angle_opts.dimension
                    << "_tp" << angle_opts.polar_points << "_" << polar_scheme
                    << "_ap" << angle_opts.azimuth_points << "_" << azimuth_scheme
                    << ".txt";
            sweep.WriteToFile(angle_quad, spatial.Mesh(), sw_path.str());
            std::cout << "Sweep order written to: " << sw_path.str() << std::endl;
        }
        catch (const std::exception &sweep_ex)
        {
            std::cerr << "Failed to build/write sweep order: " << sweep_ex.what()
                      << std::endl;
        }
    }

    // Assemble element-wise integrals to verify DG data is accessible.
    pbte::DGElementIntegrator integrator(spatial.FESpace());
    const auto element_data = integrator.AssembleAll();
    if (is_root)
    {
        std::cout << "Computed DG integrals for " << element_data.size()
                  << " elements.\n";
        if (!element_data.empty())
        {
            const auto &m0 = element_data.front().mass_matrix;
            std::cout << "Element 0 mass matrix size: " << m0.Height() << "x"
                      << m0.Width() << std::endl;

            // Sanity checks on face couplings and boundary data for element 0.
            const auto &fcs = element_data.front().face_couplings;
            std::cout << "Element 0 face couplings: " << fcs.size()
                      << " entries.\n";
            for (size_t k = 0; k < fcs.size(); ++k)
            {
                const auto &fc = fcs[k];
                std::string type = "boundary";
                if (fc.is_shared)
                {
                    type = "shared";
                }
                else if (fc.neighbor_elem >= 0)
                {
                    type = "interior";
                }
                std::cout << "  face " << fc.face_id << " " << type
                          << ", neigh=" << fc.neighbor_elem
                          << ", attr=" << fc.boundary_attr;
                if (fc.neighbor_elem >= 0 || fc.is_shared)
                {
                    std::cout << ", coupling " << fc.coupling.Height() << "x"
                              << fc.coupling.Width();
                }
                else
                {
                    std::cout << ", isothermal_rhs size "
                              << fc.isothermal_rhs.Size();
                }
                std::cout << "\n";
            }
        }
    }

    // Serialize local integrals to a string (per rank).
    std::ostringstream oss;
    oss << "DG integral dump (local rank block)\n";
#ifdef MFEM_USE_MPI
    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    oss << "rank: " << world_rank << "/" << world_size << "\n";
#else
    oss << "rank: 0/1\n";
    const int world_rank = 0;
#endif
    oss << "elements: " << element_data.size() << "\n";
    for (size_t e = 0; e < element_data.size(); ++e)
    {
        const auto &ed = element_data[e];
        oss << "\n=== Element " << e << " (rank " << world_rank << ") ===\n";
        WriteVector(oss, ed.basis_integrals, "basis_integrals");
        WriteMatrix(oss, ed.mass_matrix, "mass_matrix");
        for (size_t d = 0; d < ed.stiffness_matrices.size(); ++d)
        {
            WriteMatrix(oss, ed.stiffness_matrices[d],
                        "stiffness_matrix_dim" + std::to_string(d));
        }
        for (size_t f = 0; f < ed.face_mass_matrices.size(); ++f)
        {
            WriteMatrix(oss, ed.face_mass_matrices[f],
                        "face_mass_matrix[" + std::to_string(f) + "]");
            WriteVector(oss, ed.face_integrals[f],
                        "face_integral[" + std::to_string(f) + "]");
        }
        for (size_t k = 0; k < ed.face_couplings.size(); ++k)
        {
            const auto &fc = ed.face_couplings[k];
            oss << "face_coupling[" << k << "]: face_id=" << fc.face_id
                << ", neighbor=" << fc.neighbor_elem
                << ", attr=" << fc.boundary_attr
                << ", shared=" << (fc.is_shared ? 1 : 0) << "\n";
            if (fc.neighbor_elem >= 0 || fc.is_shared)
            {
                WriteMatrix(oss, fc.coupling, "  coupling");
            }
            else
            {
                WriteVector(oss, fc.isothermal_rhs, "  isothermal_rhs");
            }
        }
    }

#ifdef MFEM_USE_MPI
    // Gather all rank strings to root and write once.
    const std::string local_str = oss.str();
    int local_len = static_cast<int>(local_str.size());
    std::vector<int> recv_counts, displs;
    std::vector<char> recv_buf;
    if (is_root)
    {
        recv_counts.resize(world_size);
    }
    MPI_Gather(&local_len, 1, MPI_INT,
               is_root ? recv_counts.data() : nullptr, 1, MPI_INT,
               0, MPI_COMM_WORLD);
    if (is_root)
    {
        displs.resize(world_size);
        int total = 0;
        for (int i = 0; i < world_size; ++i)
        {
            displs[i] = total;
            total += recv_counts[i];
        }
        recv_buf.resize(total);
        MPI_Gatherv(local_str.data(), local_len, MPI_CHAR,
                    recv_buf.data(), recv_counts.data(), displs.data(),
                    MPI_CHAR, 0, MPI_COMM_WORLD);

        const fs::path out_path = fs::path("output/log/integrals_all.txt");
        fs::create_directories(out_path.parent_path());
        std::ofstream ofs(out_path);
        if (!ofs)
        {
            std::cerr << "Failed to open " << out_path << " for writing.\n";
            return 0;
        }
        // Write concatenated rank blocks in rank order.
        for (int r = 0; r < world_size; ++r)
        {
            ofs.write(recv_buf.data() + displs[r], recv_counts[r]);
            ofs << "\n";
        }
        ofs << std::flush;
        std::cout << "Integral dump written to: " << out_path << std::endl;
    }
    else
    {
        MPI_Gatherv(local_str.data(), local_len, MPI_CHAR,
                    nullptr, nullptr, nullptr, MPI_CHAR,
                    0, MPI_COMM_WORLD);
    }
#else
    // Serial: write directly.
    const fs::path out_path = fs::path("output/log/integrals_all.txt");
    fs::create_directories(out_path.parent_path());
    std::ofstream ofs(out_path);
    if (!ofs)
    {
        std::cerr << "Failed to open " << out_path << " for writing.\n";
        return 0;
    }
    ofs << oss.str();
    std::cout << "Integral dump written to: " << out_path << std::endl;
#endif

    return 0;
}