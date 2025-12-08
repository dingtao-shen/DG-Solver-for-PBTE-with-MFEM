#include "SpatialMesh.hpp"

#include "mfem.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

int main(int argc, char *argv[])
{
    std::string mesh_spec;  // optional override; if empty, read from config
    std::string config_path = "config/config.yaml";
    int order = 1;

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
        spatial.BuildDGSpace(order);
    }
    catch (const std::exception &ex)
    {
        std::cerr << "Error constructing mesh/FE space: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}