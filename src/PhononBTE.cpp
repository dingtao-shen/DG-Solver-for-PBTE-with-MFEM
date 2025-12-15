#include "ElementIntegrator.hpp"
#include "SpatialMesh.hpp"

#include "mfem.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace
{
void WriteVector(std::ofstream &ofs, const mfem::Vector &v,
                 const std::string &label)
{
    ofs << label << " [size=" << v.Size() << "]:";
    for (int i = 0; i < v.Size(); ++i)
    {
        ofs << " " << v[i];
    }
    ofs << "\n";
}

void WriteMatrix(std::ofstream &ofs, const mfem::DenseMatrix &m,
                 const std::string &label)
{
    ofs << label << " [shape=" << m.Height() << "x" << m.Width() << "]\n";
    for (int i = 0; i < m.Height(); ++i)
    {
        ofs << "  ";
        for (int j = 0; j < m.Width(); ++j)
        {
            ofs << m(i, j);
            if (j + 1 < m.Width())
            {
                ofs << " ";
            }
        }
        ofs << "\n";
    }
}
}  // namespace

int main(int argc, char *argv[])
{
    std::string mesh_spec;  // optional override; if empty, read from config
    std::string config_path = "config/config.yaml";
    int order = 1;
    bool use_parallel = false;

#ifdef MFEM_USE_MPI
    mfem::MPI_Session mpi(argc, argv);
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
    args.AddOption(&use_parallel,
                   "-p", "--parallel",
                   "-np", "--no-parallel",
                   "Use parallel mesh/space (requires MFEM built with MPI).",
                   false);
    args.Parse();
    if (!args.Good())
    {
        args.PrintUsage(std::cout);
        return 1;
    }
    args.PrintOptions(std::cout);

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

    // Assemble element-wise integrals to verify DG data is accessible.
    pbte::DGElementIntegrator integrator(spatial.FESpace());
    const auto element_data = integrator.AssembleAll();
    std::cout << "Computed DG integrals for " << element_data.size()
              << " elements.\n";
    if (!element_data.empty())
    {
        const auto &m0 = element_data.front().mass_matrix;
        std::cout << "Element 0 mass matrix size: " << m0.Height() << "x"
                  << m0.Width() << std::endl;

        // Sanity checks on face couplings and boundary data for element 0.
        const auto &fcs = element_data.front().face_couplings;
        std::cout << "Element 0 face couplings: " << fcs.size() << " entries.\n";
        for (size_t k = 0; k < fcs.size(); ++k)
        {
            const auto &fc = fcs[k];
            std::cout << "  face " << fc.face_id
                      << (fc.neighbor_elem >= 0 ? " interior" : " boundary")
                      << ", neigh=" << fc.neighbor_elem
                      << ", attr=" << fc.boundary_attr;
            if (fc.neighbor_elem >= 0)
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

    // Write all element integrals to disk for inspection.
    const fs::path out_path = fs::path("output/log/integrals_all.txt");
    fs::create_directories(out_path.parent_path());
    std::ofstream ofs(out_path);
    if (!ofs)
    {
        std::cerr << "Failed to open " << out_path << " for writing.\n";
        return 0;
    }

    ofs << "DG integral dump\n";
    ofs << "elements: " << element_data.size() << "\n";
    for (size_t e = 0; e < element_data.size(); ++e)
    {
        const auto &ed = element_data[e];
        ofs << "\n=== Element " << e << " ===\n";
        WriteVector(ofs, ed.basis_integrals, "basis_integrals");
        WriteMatrix(ofs, ed.mass_matrix, "mass_matrix");
        for (size_t d = 0; d < ed.stiffness_matrices.size(); ++d)
        {
            WriteMatrix(ofs, ed.stiffness_matrices[d],
                        "stiffness_matrix_dim" + std::to_string(d));
        }
        for (size_t f = 0; f < ed.face_mass_matrices.size(); ++f)
        {
            WriteMatrix(ofs, ed.face_mass_matrices[f],
                        "face_mass_matrix[" + std::to_string(f) + "]");
            WriteVector(ofs, ed.face_integrals[f],
                        "face_integral[" + std::to_string(f) + "]");
        }
        for (size_t k = 0; k < ed.face_couplings.size(); ++k)
        {
            const auto &fc = ed.face_couplings[k];
            ofs << "face_coupling[" << k << "]: face_id=" << fc.face_id
                << ", neighbor=" << fc.neighbor_elem
                << ", attr=" << fc.boundary_attr << "\n";
            if (fc.neighbor_elem >= 0)
            {
                WriteMatrix(ofs, fc.coupling, "  coupling");
            }
            else
            {
                WriteVector(ofs, fc.isothermal_rhs, "  isothermal_rhs");
            }
        }
    }

    std::cout << "Integral dump written to: " << out_path << std::endl;

    return 0;
}