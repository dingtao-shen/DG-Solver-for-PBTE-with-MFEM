// Phonon spectral properties and material parameters.
#include "PhononProperties.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace fs = std::filesystem;

namespace pbte
{
namespace
{
constexpr double PI = 3.14159265358979323846;
constexpr double H = 1.054571800e-34;   // reduced Planck [J*s]
constexpr double KB = 1.38064852e-23;   // Boltzmann [J/K]
}  // namespace

PhononMaterial PhononProperties::LoadMaterial(const std::string &path)
{
    YAML::Node config = YAML::LoadFile(path);

    PhononMaterial mat;
    mat.C_LA = config["C_LA"].as<std::vector<double>>();
    mat.C_TA = config["C_TA"].as<std::vector<double>>();
    mat.lattice_dist = config["lattice_dist"].as<double>();

    mat.num_branches = config["num_branches"].as<int>();
    mat.K_range.resize(mat.num_branches);
    mat.K_range[0] = 0.0;
    mat.K_range[1] = 2.0 * PI / mat.lattice_dist;

    mat.Ai = config["Ai"].as<double>();
    mat.BL = config["BL"].as<double>();
    mat.BT = config["BT"].as<double>();
    mat.BU = config["BU"].as<double>();

    mat.num_spectral = config["num_spectral"].as<int>();
    mat.ref_temp = config["reference_temperature"].as<double>();
    mat.ref_len = config["reference_length"].as<double>();

    return mat;
}

PhononProperties PhononProperties::Build(const PhononMaterial &mat)
{
    PhononProperties ph;
    ph.num_branches_ = mat.num_branches;
    ph.num_spectral_ = mat.num_spectral;
    ph.k_max_ = mat.K_range[1];
    ph.ref_temp_ = mat.ref_temp;
    ph.ref_len_ = mat.ref_len;

    ph.w_.assign(mat.num_branches, std::vector<double>(mat.num_spectral, 0.0));
    ph.k_.assign(mat.num_branches, std::vector<double>(mat.num_spectral, 0.0));
    ph.dw_.assign(mat.num_branches, std::vector<double>(mat.num_spectral, 0.0));
    ph.inv_kn_.assign(mat.num_branches, std::vector<double>(mat.num_spectral, 0.0));
    ph.vg_.assign(mat.num_branches, std::vector<double>(mat.num_spectral, 0.0));
    ph.density_.assign(mat.num_branches, std::vector<double>(mat.num_spectral, 0.0));
    ph.heat_cap_.assign(mat.num_branches, std::vector<double>(mat.num_spectral, 0.0));

    // Spectral grid: midpoint rule in k-space (matches reference).
    std::vector<double> kb(mat.num_spectral, 0.0);
    for (int i = 0; i < mat.num_spectral; ++i)
    {
        kb[i] = (2.0 * (i + 1) - 1.0) / (2.0 * mat.num_spectral) * mat.K_range[1];
    }

    // LA
    for (int j = 0; j < mat.num_spectral; ++j)
    {
        const double k = kb[j];
        const double w = mat.C_LA[0] * k + mat.C_LA[1] * k * k;
        const double vg = mat.C_LA[0] + 2.0 * mat.C_LA[1] * k;
        const double dw = mat.K_range[1] * vg;
        const double inv = mat.Ai * std::pow(w, 4) + mat.BL * std::pow(mat.ref_temp, 3) * std::pow(w, 2);
        const double d = k * k / vg / 2.0 / std::pow(PI, 2);

        ph.k_[0][j] = k;
        ph.w_[0][j] = w;
        ph.dw_[0][j] = dw;
        ph.vg_[0][j] = vg;
        ph.inv_kn_[0][j] = inv;
        ph.density_[0][j] = d;
    }
    // TA
    for (int j = 0; j < mat.num_spectral; ++j)
    {
        const double k = kb[j];
        const double w = mat.C_TA[0] * k + mat.C_TA[1] * k * k;
        const double vg = mat.C_TA[0] + 2.0 * mat.C_TA[1] * k;
        const double dw = mat.K_range[1] * vg;
        double inv = mat.Ai * std::pow(w, 4);
        if (k < mat.K_range[1] / 2.0)
        {
            inv += mat.BT * w * std::pow(mat.ref_temp, 4);
        }
        else
        {
            inv += mat.BU * std::pow(w, 2) /
                    std::sinh(H * w / KB / mat.ref_temp);
        }
        const double d = k * k / vg / 2.0 / std::pow(PI, 2);

        ph.k_[1][j] = k;
        ph.w_[1][j] = w;
        ph.dw_[1][j] = dw;
        ph.vg_[1][j] = vg;
        ph.inv_kn_[1][j] = inv;
        ph.density_[1][j] = d;
    }
    

    ph.heat_cap_v_ = 0.0;
    for (int p = 0; p < ph.num_branches_; ++p)
    {
        for (int j = 0; j < mat.num_spectral; ++j)
        {
            const double w = ph.w_[p][j];
            const double d = ph.density_[p][j];
            const double x = H * w / KB / mat.ref_temp;
            const double expx = std::exp(x);
            const double denom = expx - 1.0;
            const double heat_cap = std::pow(H, 2) * w * w * d * expx /
                                (denom * denom) / KB / (mat.ref_temp * mat.ref_temp);
            ph.heat_cap_[p][j] = heat_cap;
            const double w1 = mat.K_range[1] * ph.vg_[p][j];
            ph.heat_cap_v_ += heat_cap * ph.inv_kn_[p][j] * w1;
        }
    }

    return ph;
}

void PhononProperties::WriteToFile(const std::string &path) const
{
    fs::path p(path);
    if (p.has_parent_path())
    {
        fs::create_directories(p.parent_path());
    }
    std::ofstream ofs(p);
    if (!ofs)
    {
        throw std::runtime_error("Failed to open phonon log: " + path);
    }
    ofs << "Phonon properties\n";
    ofs << "num_branches: " << num_branches_ << "\n";
    ofs << "num_spectral: " << num_spectral_ << "\n";
    ofs << "k_max: " << k_max_ << "\n";
    ofs << "reference_temperature: " << ref_temp_ << "\n";
    ofs << "reference_length: " << ref_len_ << "\n";
    ofs << "HeatCapV: " << heat_cap_v_ << "\n\n";

    ofs << "branch idx k w dw vg invKn density heatCap\n";
    for (int p = 0; p < num_branches_; ++p)
    {
        for (int j = 0; j < num_spectral_; ++j)
        {
            ofs << p << " " << j << " "
                << k_[p][j] << " "
                << w_[p][j] << " "
                << dw_[p][j] << " "
                << vg_[p][j] << " "
                << inv_kn_[p][j] << " "
                << density_[p][j] << " "
                << heat_cap_[p][j] << "\n";
        }
    }
    ofs << std::flush;
}
}  // namespace pbte


