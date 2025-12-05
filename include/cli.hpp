#pragma once

#include <mfem.hpp>
#include <string>
#include "params.hpp"
#include "quadrature_sn.hpp"

namespace dg {

struct CLIConfig {
    int dim{2};
    int snOrder{2};
    double vg{1.0};
    double knN{0.2};
    double knR{0.1};
    double Lchar{1.0};
    double T{1.0};
    double ux{0.1}, uy{0.0}, uz{0.0};
    // Isothermal wall setup: one hot boundary attribute id, others cold
    double Thot{1.1};
    double Tcold{0.9};
    int hotAttr{1};
    int nx{2}, ny{2}, nz{2};
    double Lx{1.0}, Ly{1.0}, Lz{1.0};
    int p{1}; // DG polynomial order (L2 space)
    // Source iteration controls
    int maxIters{50};
    double rtol{1e-8};
    double relax{0.5};
    // Output
    bool saveVTK{true};
    std::string outPrefix{"dg4pbte_out"};
    // Toggle isothermal inflow BC (experimental integrator)
    bool useIsoBC{false};
};

inline void ParseCLI(int argc, char** argv, CLIConfig& cfg)
{
    mfem::OptionsParser p(argc, argv);
    p.AddOption(&cfg.dim, "-d", "--dim", "Spatial dimension (2 or 3).");
    p.AddOption(&cfg.snOrder, "-s", "--sn", "SN order (small integer).");
    p.AddOption(&cfg.vg, "-vg", "--group-velocity", "Group velocity (gray).");
    p.AddOption(&cfg.knN, "-knN", "--knudsen-normal", "Knudsen number for normal scattering.");
    p.AddOption(&cfg.knR, "-knR", "--knudsen-resistive", "Knudsen number for resistive scattering.");
    p.AddOption(&cfg.Lchar, "-L", "--length", "Characteristic length for tau computation.");
    p.AddOption(&cfg.T, "-T", "--temperature", "Reference temperature.");
    p.AddOption(&cfg.ux, "-ux", "--drift-x", "Drift x.");
    p.AddOption(&cfg.uy, "-uy", "--drift-y", "Drift y.");
    p.AddOption(&cfg.uz, "-uz", "--drift-z", "Drift z.");
    p.AddOption(&cfg.Thot, "-Thot", "--T-hot", "Hot wall temperature.");
    p.AddOption(&cfg.Tcold, "-Tcold", "--T-cold", "Cold wall temperature.");
    p.AddOption(&cfg.hotAttr, "-hot", "--hot-attr", "Boundary attribute id for hot wall.");
    p.AddOption(&cfg.nx, "-nx", "--nx", "Mesh cells in x.");
    p.AddOption(&cfg.ny, "-ny", "--ny", "Mesh cells in y.");
    p.AddOption(&cfg.nz, "-nz", "--nz", "Mesh cells in z.");
    p.AddOption(&cfg.Lx, "-Lx", "--Lx", "Domain length in x.");
    p.AddOption(&cfg.Ly, "-Ly", "--Ly", "Domain length in y.");
    p.AddOption(&cfg.Lz, "-Lz", "--Lz", "Domain length in z.");
    p.AddOption(&cfg.p, "-p", "--order", "DG polynomial order.");
    p.AddOption(&cfg.maxIters, "-maxit", "--max-iters", "Max source iterations.");
    p.AddOption(&cfg.rtol, "-rtol", "--rel-tol", "Relative tolerance for source iteration.");
    p.AddOption(&cfg.relax, "-relax", "--relaxation", "Under-relaxation for macro updates (0,1].");
    // For boolean flags, use enable/disable forms
    p.AddOption(&cfg.saveVTK, "-vtk", "--save-vtk",
                "-no-vtk", "--no-save-vtk",
                "Save ParaView VTK output.");
    p.AddOption(&cfg.outPrefix, "-o", "--output-prefix", "Output prefix for files.");
    p.AddOption(&cfg.useIsoBC, "-bc", "--isothermal-bc",
                "-no-bc", "--no-isothermal-bc",
                "Enable isothermal inflow BC (experimental).");
    p.Parse();
    if (!p.Good()) { p.PrintUsage(std::cout); exit(1); }
}

inline GrayCallawayParams MakeParams(const CLIConfig& cfg)
{
    GrayCallawayParams params;
    params.dimension = cfg.dim;
    params.groupVelocity = cfg.vg;
    params.knudsenNormal = cfg.knN;
    params.knudsenResistive = cfg.knR;
    return params;
}

inline SNDirections MakeSN(const CLIConfig& cfg)
{
    return makeLevelSymmetricSN(cfg.snOrder, cfg.dim);
}

} // namespace dg

