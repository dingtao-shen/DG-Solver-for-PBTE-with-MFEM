// Phonon spectral properties and material parameters.
#pragma once

#include <string>
#include <vector>

namespace pbte
{
struct PhononMaterial
{
    // Dispersion coefficients: w = c0 * k + c1 * k^2.
    int num_branches = 0;
    std::vector<double> C_LA;
    std::vector<double> C_TA;
    double lattice_dist = 0.0;  // [m]
    std::vector<double> K_range;

    // Scattering parameters.
    double Ai = 0.0;
    double BL = 0.0;
    double BT = 0.0;
    double BU = 0.0;

    // Spectral/grid options.
    int num_spectral = 0;        // NSPEC
    double ref_temp = 0.0;  // [K]
    double ref_len = 0.0;  // [m]
};

/// Frequency/velocity/scattering tables for LA/TA branches.
class PhononProperties
{
public:
    /// Load material parameters from YAML-like file (config/si.yaml style).
    static PhononMaterial LoadMaterial(const std::string &path);

    /// Build spectral properties using material params and options.
    static PhononProperties Build(const PhononMaterial &mat);

    /// Write summary and tables to a file.
    void WriteToFile(const std::string &path) const;

    // Accessors (branch index matches input ordering: 0=LA, 1=TA).
    const std::vector<std::vector<double>> &WaveVector() const { return k_; }
    const std::vector<double> &WaveVector(int branch) const { return k_[branch]; }

    const std::vector<std::vector<double>> &Frequency() const { return w_; }
    const std::vector<double> &Frequency(int branch) const { return w_[branch]; }

    const std::vector<std::vector<double>> &FrequencyWeight() const { return dw_; }
    const std::vector<double> &FrequencyWeight(int branch) const { return dw_[branch]; }

    const std::vector<std::vector<double>> &GroupVelocity() const { return vg_; }
    const std::vector<double> &GroupVelocity(int branch) const { return vg_[branch]; }

    const std::vector<std::vector<double>> &InvKn() const { return inv_kn_; }
    const std::vector<double> &InvKn(int branch) const { return inv_kn_[branch]; }

    const std::vector<std::vector<double>> &DensityOfStates() const { return density_; }
    const std::vector<double> &DensityOfStates(int branch) const { return density_[branch]; }

    const std::vector<std::vector<double>> &HeatCapacity() const { return heat_cap_; }
    const std::vector<double> &HeatCapacity(int branch) const { return heat_cap_[branch]; }

    double avgHeatCapacity() const { return heat_cap_v_; }

private:
    int num_branches_ = 0;
    int num_spectral_ = 0;
    double k_max_ = 0.0;
    double ref_temp_ = 0.0;
    double ref_len_ = 0.0;
    std::vector<std::vector<double>> k_;       // wave vector
    std::vector<std::vector<double>> w_;       // frequency
    std::vector<std::vector<double>> dw_;       // frequency weight
    std::vector<std::vector<double>> vg_;      // group velocity
    std::vector<std::vector<double>> inv_kn_;  // inverse Kn
    std::vector<std::vector<double>> density_; // state density factor
    std::vector<std::vector<double>> heat_cap_;
    double heat_cap_v_ = 0.0;
};
}  // namespace pbte


