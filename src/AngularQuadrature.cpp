// Angular quadrature (solid-angle discretization) utilities.
#include "AngularQuadrature.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace fs = std::filesystem;

namespace pbte
{
namespace
{
constexpr double kPi = 3.14159265358979323846;

std::string Trim(const std::string &s)
{
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
    {
        return "";
    }
    const auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

std::string ToLower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::vector<std::pair<double, double>> UniformMidpointRule(int points,
                                                           double a,
                                                           double b)
{
    if (points <= 0)
    {
        throw std::invalid_argument("Uniform rule requires positive point count.");
    }
    const double h = (b - a) / static_cast<double>(points);
    std::vector<std::pair<double, double>> nodes;
    nodes.reserve(points);
    for (int i = 0; i < points; ++i)
    {
        const double x = a + (i + 0.5) * h;
        nodes.emplace_back(x, h);
    }
    return nodes;
}

std::vector<std::pair<double, double>> GaussLegendreRule(int points,
                                                         double a,
                                                         double b)
{
    if (points <= 0)
    {
        throw std::invalid_argument("Gauss-Legendre rule requires positive point count.");
    }

    // MFEM integration rules: order = 2*n - 1 for Gauss-Legendre with n points.
    const int order = 2 * points - 1;
    mfem::IntegrationRules rules(0, mfem::Quadrature1D::GaussLegendre);
    const mfem::IntegrationRule &ir =
        rules.Get(mfem::Geometry::SEGMENT, order);

    std::vector<std::pair<double, double>> nodes;
    nodes.reserve(points);
    // MFEM's 1D integration points/weights are defined on the reference
    // segment [-1, 1]. Map them affinely to [a, b].
    //
    // Note: the previous implementation incorrectly treated ip.x as living
    // on [0, 1], which shifted nodes outside the target interval and doubled
    // weights. Here we use the standard affine map from [-1, 1] to [a, b].
    const double half = 0.5 * (b - a);
    const double mid = 0.5 * (b + a);
    for (int i = 0; i < ir.GetNPoints(); ++i)
    {
        const mfem::IntegrationPoint &ip = ir.IntPoint(i);
        const double x = mid + half * ip.x;
        const double w = half * ip.weight;
        nodes.emplace_back(x, w);
    }
    return nodes;
}

AngleDiscretizationType ParseSchemeInternal(const std::string &name)
{
    const std::string key = ToLower(Trim(name));
    if (key == "uniform")
    {
        return AngleDiscretizationType::Uniform;
    }
    if (key == "gauss" || key == "gauss-legendre" || key == "legendre")
    {
        return AngleDiscretizationType::GaussLegendre;
    }
    throw std::invalid_argument("Unknown discretization scheme: " + name);
}

std::vector<std::string> LoadConfigLines(const std::string &config_path)
{
    std::ifstream in(config_path);
    if (!in)
    {
        throw std::runtime_error("Failed to open config file: " + config_path);
    }
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
    return lines;
}
}  // namespace

AngleDiscretizationType AngleQuadrature::ParseSchemeName(const std::string &name)
{
    return ParseSchemeInternal(name);
}

std::string AngleQuadrature::SchemeName(AngleDiscretizationType type)
{
    switch (type)
    {
    case AngleDiscretizationType::Uniform:
        return "uniform";
    case AngleDiscretizationType::GaussLegendre:
        return "gauss";
    }
    return "unknown";
}

AngleDiscretizationOptions AngleQuadrature::LoadOptionsFromConfig(
    const std::string &config_path)
{
    AngleDiscretizationOptions opts;
    const auto lines = LoadConfigLines(config_path);

    bool in_angles_block = false;
    for (const auto &raw : lines)
    {
        const std::string l = Trim(raw);
        if (l.empty())
        {
            continue;
        }
        if (l.rfind("angles:", 0) == 0)
        {
            in_angles_block = true;
            continue;
        }
        if (!in_angles_block)
        {
            continue;
        }

        if (l.rfind("dimension:", 0) == 0)
        {
            opts.dimension = std::stoi(Trim(l.substr(std::string("dimension:").size())));
        }
        else if (l.rfind("polar_points:", 0) == 0)
        {
            opts.polar_points =
                std::stoi(Trim(l.substr(std::string("polar_points:").size())));
        }
        else if (l.rfind("azimuth_points:", 0) == 0)
        {
            opts.azimuth_points =
                std::stoi(Trim(l.substr(std::string("azimuth_points:").size())));
        }
        else if (l.rfind("polar_scheme:", 0) == 0)
        {
            opts.polar_scheme =
                ParseSchemeInternal(Trim(l.substr(std::string("polar_scheme:").size())));
        }
        else if (l.rfind("azimuth_scheme:", 0) == 0)
        {
            opts.azimuth_scheme = ParseSchemeInternal(
                Trim(l.substr(std::string("azimuth_scheme:").size())));
        }
    }

    return opts;
}

AngleQuadrature AngleQuadrature::Build(const AngleDiscretizationOptions &opts)
{
    AngleQuadrature quad;
    if (opts.dimension != 2 && opts.dimension != 3)
    {
        throw std::invalid_argument("Angular quadrature dimension must be 2 or 3.");
    }
    quad.dimension_ = opts.dimension;

    // Polar nodes: in 3D discretize mu = cos(theta) in [-1, 1]; in 2D use a
    // single polar node at theta = pi/2 with unit weight (all in-plane).
    std::vector<std::pair<double, double>> polar_nodes;
    if (opts.dimension == 2)
    {
        polar_nodes.emplace_back(0.0, 1.0);  // mu=cos(theta)=0
    }
    else
    {
        if (opts.polar_points <= 0)
        {
            throw std::invalid_argument("polar_points must be positive for 3D.");
        }
        if (opts.polar_scheme == AngleDiscretizationType::Uniform)
        {
            polar_nodes = UniformMidpointRule(opts.polar_points, -1.0, 1.0);
        }
        else
        {
            polar_nodes = GaussLegendreRule(opts.polar_points, -1.0, 1.0);
        }
    }

    quad.polar_angles_.reserve(polar_nodes.size());
    quad.polar_weights_.reserve(polar_nodes.size());
    for (const auto &[mu, w] : polar_nodes)
    {
        const double clamped_mu = std::max(-1.0, std::min(1.0, mu));
        const double theta = std::acos(clamped_mu);
        quad.polar_angles_.push_back(theta);
        quad.polar_weights_.push_back(w);
    }

    // Azimuthal nodes: phi in [0, 2*pi].
    if (opts.azimuth_points <= 0)
    {
        throw std::invalid_argument("azimuth_points must be positive.");
    }
    std::vector<std::pair<double, double>> az_nodes;
    if (opts.azimuth_scheme == AngleDiscretizationType::Uniform)
    {
        az_nodes = UniformMidpointRule(opts.azimuth_points, 0.0, 2.0 * kPi);
    }
    else
    {
        az_nodes = GaussLegendreRule(opts.azimuth_points, 0.0, 2.0 * kPi);
    }
    quad.azimuth_angles_.reserve(az_nodes.size());
    quad.azimuth_weights_.reserve(az_nodes.size());
    for (const auto &[phi, w] : az_nodes)
    {
        quad.azimuth_angles_.push_back(phi);
        quad.azimuth_weights_.push_back(w);
    }

    // Build tensor-product directions.
    for (size_t it = 0; it < quad.polar_angles_.size(); ++it)
    {
        const double theta = quad.polar_angles_[it];
        const double w_theta = quad.polar_weights_[it];
        const double sin_theta = std::sin(theta);
        const double cos_theta = std::cos(theta);

        for (size_t ip = 0; ip < quad.azimuth_angles_.size(); ++ip)
        {
            const double phi = quad.azimuth_angles_[ip];
            const double w_phi = quad.azimuth_weights_[ip];

            AngleDirection dir;
            dir.polar = theta;
            dir.azimuth = phi;
            dir.weight = w_theta * w_phi;
            dir.direction[0] = sin_theta * std::cos(phi);
            dir.direction[1] = sin_theta * std::sin(phi);
            dir.direction[2] = (opts.dimension == 3) ? cos_theta : 0.0;

            quad.total_weight_ += dir.weight;
            quad.directions_.push_back(std::move(dir));
        }
    }

    // Normalize total weight to the exact solid-angle measure (2*pi in 2D, 4*pi in 3D).
    const double expected_total = (opts.dimension == 3) ? 4.0 * kPi : 2.0 * kPi;
    if (quad.total_weight_ > 0.0)
    {
        const double scale = expected_total / quad.total_weight_;
        for (auto &d : quad.directions_)
        {
            d.weight *= scale;
        }
        quad.total_weight_ = expected_total;
    }

    return quad;
}

void AngleQuadrature::WriteToFile(const std::string &path) const
{
    fs::path p(path);
    if (p.has_parent_path())
    {
        fs::create_directories(p.parent_path());
    }
    std::ofstream ofs(p);
    if (!ofs)
    {
        throw std::runtime_error("Failed to open angle log file: " + path);
    }

    const size_t ntheta = polar_angles_.size();
    const size_t nphi = azimuth_angles_.size();
    ofs << "Angular quadrature summary\n";
    ofs << "  dimension        : " << dimension_ << "\n";
    ofs << "  polar points     : " << ntheta << "\n";
    ofs << "  azimuth points   : " << nphi << "\n";
    ofs << "  directions       : " << directions_.size() << "\n";
    ofs << "  total weight     : " << total_weight_ << "\n\n";

    ofs << "Directions (idx, theta, phi, weight, dir_x, dir_y, dir_z)\n";
    for (size_t i = 0; i < directions_.size(); ++i)
    {
        const auto &d = directions_[i];
        ofs << i << " "
            << d.polar << " "
            << d.azimuth << " "
            << d.weight << " "
            << d.direction[0] << " "
            << d.direction[1] << " "
            << d.direction[2] << "\n";
    }
}
}  // namespace pbte


