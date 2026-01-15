// Compute element sweep order for each angular direction.
#include "AngularSweepOrder.hpp"
#include "Utils.hpp"

#include <filesystem>
#include <fstream>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>

namespace fs = std::filesystem;

namespace pbte
{
namespace
{
mfem::Vector ComputeCentroid(const mfem::Mesh &mesh, int elem_id)
{
    mfem::Array<int> verts;
    mesh.GetElementVertices(elem_id, verts);
    const int dim = mesh.Dimension();
    mfem::Vector c(dim);
    c = 0.0;
    for (int vid : verts)
    {
        const double *v = mesh.GetVertex(vid);
        for (int d = 0; d < dim; ++d)
        {
            c[d] += v[d];
        }
    }
    if (verts.Size() > 0)
    {
        c /= static_cast<double>(verts.Size());
    }
    return c;
}

struct NeighborInfo
{
    int neighbor = -1;   // -1 for boundary
    mfem::Vector normal; // outward from the owning element
};

// Build adjacency: for each element, list of neighbors with outward normals.
std::vector<std::vector<NeighborInfo>> BuildAdjacency(const mfem::Mesh &mesh)
{
    const int ne = mesh.GetNE();
    std::vector<std::vector<NeighborInfo>> adj(ne);

    const int nfaces = mesh.GetNumFaces();
    for (int f = 0; f < nfaces; ++f)
    {
        int e1 = -1, e2 = -1;
        mesh.GetFaceElements(f, &e1, &e2);

        auto add_neighbor = [&](int owner, int neigh) {
            if (owner < 0)
            {
                return;
            }
            NeighborInfo info;
            info.neighbor = neigh;
            info.normal.SetSize(mesh.SpaceDimension());
            info.normal = utils::ComputeOutwardFaceNormal(mesh, f, owner);
            adj[owner].push_back(std::move(info));
        };

        add_neighbor(e1, e2);
        add_neighbor(e2, e1);
    }

    return adj;
}
}  // namespace

AngularSweepOrder AngularSweepOrder::Build(const mfem::Mesh &mesh,
                                           const AngleQuadrature &angle_quad)
{
    const int ne = mesh.GetNE();
    const int ndir = static_cast<int>(angle_quad.Directions().size());
    if (ne == 0 || ndir == 0)
    {
        return AngularSweepOrder{};
    }

    AngularSweepOrder result;
    result.orders_.assign(ndir, std::vector<int>(ne, -1));

    const auto adjacency = BuildAdjacency(mesh);

    for (int k = 0; k < ndir; ++k)
    {
        const auto &dir = angle_quad.Directions()[k].direction;
        mfem::Vector d(mesh.SpaceDimension());
        for (int i = 0; i < mesh.SpaceDimension(); ++i)
        {
            d[i] = dir[i];
        }

        std::vector<bool> processed(ne, false);
        int count = 0;
        while (count < ne)
        {
            bool progressed = false;
            for (int e = 0; e < ne; ++e)
            {
                if (processed[e])
                {
                    continue;
                }
                bool ready = true;
                for (const auto &nb : adjacency[e])
                {
                    if (nb.neighbor < 0)
                    {
                        continue;  // boundary
                    }
                    if (processed[nb.neighbor])
                    {
                        continue;
                    }
                    const double dot = nb.normal * d;
                    if (dot < 0.0)
                    {
                        ready = false;
                        break;
                    }
                }
                if (ready)
                {
                    result.orders_[k][count++] = e;
                    processed[e] = true;
                    progressed = true;
                }
            }
            if (!progressed)
            {
                throw std::runtime_error(
                    "Angular sweep ordering stalled; check mesh connectivity.");
            }
        }
    }

    return result;
}

void AngularSweepOrder::WriteToFile(const AngleQuadrature &angle_quad,
                                    const mfem::Mesh &mesh,
                                    const std::string &path) const
{
    fs::path p(path);
    if (p.has_parent_path())
    {
        fs::create_directories(p.parent_path());
    }
    std::ofstream ofs(p);
    if (!ofs)
    {
        throw std::runtime_error("Failed to open sweep log: " + path);
    }
    ofs << "Sweep order per direction\n";
    ofs << "dimension: " << mesh.Dimension() << "\n";
    ofs << "elements: " << mesh.GetNE() << "\n";
    ofs << "directions: " << angle_quad.Directions().size() << "\n\n";
    for (int k = 0; k < NumDirections(); ++k)
    {
        const auto &d = angle_quad.Directions()[k];
        ofs << "dir " << k << " theta=" << d.polar
            << " phi=" << d.azimuth
            << " w=" << d.weight
            << " order:";
        for (int e : orders_[k])
        {
            ofs << " " << e;
        }
        ofs << "\n";
    }
    ofs << std::flush;
}
}  // namespace pbte


