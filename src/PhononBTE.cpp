#include "AngularQuadrature.hpp"
#include "AngularSweepOrder.hpp"
#include "ElementIntegrator.hpp"
#include "MacroscopicQuantities.hpp"
#include "PhononProperties.hpp"
#include "PBTESolver.hpp"
#include "SpatialMesh.hpp"
#include "Utils.hpp"

#include "mfem.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

int main(int argc, char *argv[])
{
    std::string mesh_spec;  // optional override; if empty, read from config
    std::string config_path = "config/config.yaml";
    int order = 1;
    int refine_levels = 0;
    bool use_parallel = false;
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
        try
        {
            const auto material = pbte::PhononProperties::LoadMaterial("config/si.yaml");
            // Scale non-dimensional mesh coordinates using reference_length from si.yaml.
            // Example: unit-square mesh (0..1) becomes (0..reference_length) in meters.
            spatial.ScaleCoordinates(material.ref_len);
        }
        catch (const std::exception &scale_ex)
        {
            if (is_root)
            {
                std::cerr << "Warning: failed to scale mesh by si.yaml reference_length: "
                          << scale_ex.what() << std::endl;
            }
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
                 << "_np" << angle_opts.polar_points << "_" << polar_scheme
                 << "_na" << angle_opts.azimuth_points << "_" << azimuth_scheme
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
                    << "_np" << angle_opts.polar_points << "_" << polar_scheme
                    << "_na" << angle_opts.azimuth_points << "_" << azimuth_scheme
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

    // Build phonon properties from material config.
    if (is_root)
    {
        try
        {
            const auto material = pbte::PhononProperties::LoadMaterial("config/si.yaml");
            const auto props = pbte::PhononProperties::Build(material);
            const std::string ph_path = "output/log/phonon_properties.txt";
            props.WriteToFile(ph_path);
            std::cout << "Phonon properties written to: " << ph_path << std::endl;
        }
        catch (const std::exception &ph_ex)
        {
            std::cerr << "Failed to build/write phonon properties: " << ph_ex.what()
                      << std::endl;
        }
    }

    // Build phonon properties for solver use (all ranks).
    pbte::PhononProperties props;
    try
    {
        const auto material = pbte::PhononProperties::LoadMaterial("config/si.yaml");
        props = pbte::PhononProperties::Build(material);
    }
    catch (const std::exception &ph_ex)
    {
        std::cerr << "Failed to build phonon properties: " << ph_ex.what() << std::endl;
        return 1;
    }

    pbte::MacroscopicQuantities macro(spatial.FESpace(), props, angle_quad);
    macro.Reset();

    // Debug: print isothermal BC map and all boundary faces carrying them
    const auto &bc_map = spatial.IsothermalBoundaryTemps();
    if (is_root)
    {
        std::cout << "Isothermal BC count = " << bc_map.size() << std::endl;
        for (const auto &kv : bc_map)
        {
            std::cout << "  attr " << kv.first << " -> T = " << kv.second << std::endl;
        }
        const mfem::Mesh &m = spatial.Mesh();
        const int nbe = m.GetNBE();
        std::cout << "Boundary faces with isothermal attributes:" << std::endl;
        for (int be = 0; be < nbe; ++be)
        {
            const int face_id = m.GetBdrElementFaceIndex(be);
            const int attr = m.GetBdrAttribute(be);
            if (!bc_map.count(attr))
            {
                continue;
            }
            mfem::Array<int> fv;
            m.GetFaceVertices(face_id, fv);
            std::cout << "  be " << be << " face " << face_id << " attr=" << attr
                      << " verts=";
            for (int i = 0; i < fv.Size(); ++i)
            {
                std::cout << fv[i] << (i + 1 < fv.Size() ? "," : "");
            }
            std::cout << std::endl;
        }
    }

#ifdef MFEM_USE_MPI
    // if (use_parallel)
    // {
    //     pbte::AngularSweepOrder sweep = pbte::AngularSweepOrder::Build(spatial.Mesh(), angle_quad);
    //     pbte::PBTESolverPar solver(*spatial.ParMeshPtr(),
    //                                *spatial.ParFESpacePtr(),
    //                                angle_quad,
    //                                sweep,
    //                                element_data,
    //                                props,
    //                                spatial.IsothermalBoundaryTemps(),
    //                                pbte::CachePolicy::FullLU,
    //                                1e-8,
    //                                100); // 100-step test

    //     const int ndir = static_cast<int>(angle_quad.Directions().size());
    //     const int nbranch = static_cast<int>(props.Frequency().size());
    //     const int nspec = nbranch > 0 ? static_cast<int>(props.Frequency(0).size()) : 0;
    //     const int ndof = spatial.FESpace().GetFE(0)->GetDof();
    //     const int ne_local = spatial.Mesh().GetNE();
    //     std::vector<std::vector<std::vector<mfem::DenseMatrix>>> coeff(
    //         ndir, std::vector<std::vector<mfem::DenseMatrix>>(
    //                   nbranch, std::vector<mfem::DenseMatrix>(
    //                                nspec, mfem::DenseMatrix(ndof, ne_local))));

    //     double res = solver.Solve(coeff, macro);
    //     if (is_root)
    //     {
    //         std::cout << "[Parallel] final residual = " << res << std::endl;
    //     }

    //     // macro.WriteParaView("pbte_fields");
    //     // For debugging/validation, export a sampled 2D temperature slice instead of ParaView.
    //     // (Only meaningful for 2D meshes.)
    //     if (spatial.Mesh().SpaceDimension() == 2)
    //     {
    //         macro.Write2DSliceTemperature("output/2D/results/T_slice.txt", 100, 100);
    //     }
    //     return 0;
    // }
#endif

    // Serial path
    {
        pbte::AngularSweepOrder sweep = pbte::AngularSweepOrder::Build(spatial.Mesh(), angle_quad);
        pbte::PBTESolver solver(spatial.Mesh(),
                                spatial.FESpace(),
                                angle_quad,
                                sweep,
                                element_data,
                                props,
                                spatial.IsothermalBoundaryTemps(),
                                pbte::CachePolicy::FullLU,
                                1e-7,
                                1); // 1 million-step test

        auto coeff = solver.CreateInitialCoefficients();

        double res = solver.Solve(coeff, macro);
        if (is_root)
        {
            std::cout << "[Serial] final residual = " << res << std::endl;
        }

        macro.WriteParaView("pbte_fields");
        // For debugging/validation, export a sampled 2D temperature slice instead of ParaView.
        // (Only meaningful for 2D meshes.)
        if (spatial.Mesh().SpaceDimension() == 2)
        {
            macro.Write2DSliceTemperature("output/2D/results/T_slice.txt", 100, 100);
        }
    }

    if (!utils::DumpElementIntegrals(element_data, is_root))
    {
        return 0;
    }

    return 0;
}