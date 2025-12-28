// Compute element sweep order for each angular direction.
#include "AngularSweepOrder.hpp"

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

mfem::Vector ComputeFaceCentroid(const mfem::Mesh &mesh, int face_id)
{
    mfem::Array<int> verts;
    mesh.GetFaceVertices(face_id, verts);
    const int dim = mesh.SpaceDimension();
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

mfem::Vector Cross3D(const mfem::Vector &a, const mfem::Vector &b)
{
    mfem::Vector c(3);
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
    return c;
}

mfem::Vector ComputeFaceNormal(const mfem::Mesh &mesh, int face_id)
{
    mfem::Array<int> verts;
    mesh.GetFaceVertices(face_id, verts);
    const int sdim = mesh.SpaceDimension();

    mfem::Vector n(sdim);
    n = 0.0;

    if (sdim == 2)
    {
        if (verts.Size() >= 2)
        {
            const double *v0 = mesh.GetVertex(verts[0]);
            const double *v1 = mesh.GetVertex(verts[1]);
            const double dx = v1[0] - v0[0];
            const double dy = v1[1] - v0[1];
            n[0] = dy;
            n[1] = -dx;
        }
    }
    else if (sdim == 3)
    {
        if (verts.Size() >= 3)
        {
            const double *v0 = mesh.GetVertex(verts[0]);
            const double *v1 = mesh.GetVertex(verts[1]);
            const double *v2 = mesh.GetVertex(verts[2]);
            mfem::Vector e1(3), e2(3);
            e1[0] = v1[0] - v0[0];
            e1[1] = v1[1] - v0[1];
            e1[2] = v1[2] - v0[2];
            e2[0] = v2[0] - v0[0];
            e2[1] = v2[1] - v0[1];
            e2[2] = v2[2] - v0[2];
            mfem::Vector c = Cross3D(e1, e2);
            n = c;
        }
    }

    const double norm = n.Norml2();
    if (norm > 0.0)
    {
        n /= norm;
    }
    return n;
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

    // Precompute element centroids.
    std::vector<mfem::Vector> elem_centroids;
    elem_centroids.reserve(ne);
    for (int e = 0; e < ne; ++e)
    {
        elem_centroids.push_back(ComputeCentroid(mesh, e));
    }

    const int nfaces = mesh.GetNumFaces();
    for (int f = 0; f < nfaces; ++f)
    {
        int e1 = -1, e2 = -1;
        mesh.GetFaceElements(f, &e1, &e2);
        const mfem::Vector face_c = ComputeFaceCentroid(mesh, f);
        const mfem::Vector face_n = ComputeFaceNormal(mesh, f);

        auto add_neighbor = [&](int owner, int neigh) {
            if (owner < 0)
            {
                return;
            }
            NeighborInfo info;
            info.neighbor = neigh;
            info.normal.SetSize(mesh.SpaceDimension());
            info.normal = face_n;
            // Ensure outward relative to owner: flip if pointing into element.
            mfem::Vector to_face = face_c;
            to_face -= elem_centroids[owner];
            if (info.normal * to_face < 0.0)
            {
                info.normal *= -1.0;
            }
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


