// Angular quadrature (solid-angle discretization) utilities.
#pragma once

#include <array>
#include <string>
#include <vector>

namespace pbte
{
/// Discretization scheme for 1D angular factors.
enum class AngleDiscretizationType
{
    Uniform,
    GaussLegendre
};

/// User-facing options for building a polar/azimuthal product rule.
struct AngleDiscretizationOptions
{
    /// Spatial dimension of the problem: 2 (in-plane) or 3.
    int dimension = 3;
    /// Number of polar (theta) points. For 2D, a single polar point is used.
    int polar_points = 8;
    /// Number of azimuthal (phi) points.
    int azimuth_points = 16;
    /// Discretization rule for polar angles (cos(theta) in [-1, 1]).
    AngleDiscretizationType polar_scheme = AngleDiscretizationType::GaussLegendre;
    /// Discretization rule for azimuthal angles (phi in [0, 2*pi]).
    AngleDiscretizationType azimuth_scheme = AngleDiscretizationType::Uniform;
};

/// One discrete solid-angle direction.
struct AngleDirection
{
    double polar = 0.0;     ///< theta in [0, pi]
    double azimuth = 0.0;   ///< phi in [0, 2*pi]
    double weight = 0.0;    ///< solid-angle weight
    std::array<double, 3> direction{0.0, 0.0, 0.0};  ///< unit vector
};

/// Product quadrature over polar/azimuth angles.
class AngleQuadrature
{
public:
    /// Build from explicit options (throws on invalid input).
    static AngleQuadrature Build(const AngleDiscretizationOptions &opts);

    /// Load options from a YAML config file. If the `angles` block is missing,
    /// defaults are returned. Throws if the file cannot be opened or if any
    /// provided value is invalid.
    static AngleDiscretizationOptions LoadOptionsFromConfig(
        const std::string &config_path);

    /// Parse scheme name (case-insensitive, accepts "uniform", "gauss",
    /// "gauss-legendre", "legendre").
    static AngleDiscretizationType ParseSchemeName(const std::string &name);
    /// Pretty-print scheme name.
    static std::string SchemeName(AngleDiscretizationType type);

    /// Write summary and all directions to the given file path.
    void WriteToFile(const std::string &path) const;

    int Dimension() const { return dimension_; }
    double TotalWeight() const { return total_weight_; }
    const std::vector<double> &PolarAngles() const { return polar_angles_; }
    const std::vector<double> &PolarWeights() const { return polar_weights_; }
    const std::vector<double> &AzimuthAngles() const { return azimuth_angles_; }
    const std::vector<double> &AzimuthWeights() const { return azimuth_weights_; }
    const std::vector<AngleDirection> &Directions() const { return directions_; }

private:
    int dimension_ = 0;
    std::vector<double> polar_angles_;
    std::vector<double> polar_weights_;
    std::vector<double> azimuth_angles_;
    std::vector<double> azimuth_weights_;
    std::vector<AngleDirection> directions_;
    double total_weight_ = 0.0;
};
}  // namespace pbte


