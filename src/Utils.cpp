#include "Utils.hpp"
#include "ElementIntegrator.hpp"
#include "AngularQuadrature.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif

namespace utils
{
namespace fs = std::filesystem;

void WriteVector(std::ostream &os, const mfem::Vector &v,
                 const std::string &label)
{
    os << label << " [size=" << v.Size() << "]:";
    for (int i = 0; i < v.Size(); ++i)
    {
        os << " " << v[i];
    }
    os << "\n";
}

void WriteMatrix(std::ostream &os, const mfem::DenseMatrix &m,
                 const std::string &label)
{
    os << label << " [shape=" << m.Height() << "x" << m.Width() << "]\n";
    for (int i = 0; i < m.Height(); ++i)
    {
        os << "  ";
        for (int j = 0; j < m.Width(); ++j)
        {
            os << m(i, j);
            if (j + 1 < m.Width())
            {
                os << " ";
            }
        }
        os << "\n";
    }
}

bool DumpElementIntegrals(const std::vector<pbte::ElementIntegralData> &element_data,
                          bool is_root)
{
    std::ostringstream oss;
    oss << "DG integral dump (local rank block)\n";
#ifdef MFEM_USE_MPI
    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    oss << "rank: " << world_rank << "/" << world_size << "\n";
#else
    oss << "rank: 0/1\n";
    const int world_rank = 0;
    const int world_size = 1;
#endif
    oss << "elements: " << element_data.size() << "\n";
    for (size_t e = 0; e < element_data.size(); ++e)
    {
        const auto &ed = element_data[e];
        oss << "\n=== Element " << e << " (rank " << world_rank << ") ===\n";
        WriteVector(oss, ed.basis_integrals, "basis_integrals");
        WriteMatrix(oss, ed.mass_matrix, "mass_matrix");
        for (size_t d = 0; d < ed.stiffness_matrices.size(); ++d)
        {
            WriteMatrix(oss, ed.stiffness_matrices[d],
                        "stiffness_matrix_dim" + std::to_string(d));
        }
        for (size_t f = 0; f < ed.face_mass_matrices.size(); ++f)
        {
            WriteMatrix(oss, ed.face_mass_matrices[f],
                        "face_mass_matrix[" + std::to_string(f) + "]");
            WriteVector(oss, ed.face_integrals[f],
                        "face_integral[" + std::to_string(f) + "]");
        }
        for (size_t k = 0; k < ed.face_couplings.size(); ++k)
        {
            const auto &fc = ed.face_couplings[k];
            oss << "face_coupling[" << k << "]: face_id=" << fc.face_id
                << ", neighbor=" << fc.neighbor_elem
                << ", attr=" << fc.boundary_attr
                << ", shared=" << (fc.is_shared ? 1 : 0) << "\n";
            if (fc.neighbor_elem >= 0 || fc.is_shared)
            {
                WriteMatrix(oss, fc.coupling, "  coupling");
            }
            else
            {
                WriteVector(oss, fc.isothermal_rhs, "  isothermal_rhs");
            }
        }
    }

#ifdef MFEM_USE_MPI
    const std::string local_str = oss.str();
    int local_len = static_cast<int>(local_str.size());
    std::vector<int> recv_counts, displs;
    std::vector<char> recv_buf;
    if (is_root)
    {
        recv_counts.resize(world_size);
    }
    MPI_Gather(&local_len, 1, MPI_INT,
               is_root ? recv_counts.data() : nullptr, 1, MPI_INT,
               0, MPI_COMM_WORLD);
    if (is_root)
    {
        displs.resize(world_size);
        int total = 0;
        for (int i = 0; i < world_size; ++i)
        {
            displs[i] = total;
            total += recv_counts[i];
        }
        recv_buf.resize(total);
        MPI_Gatherv(local_str.data(), local_len, MPI_CHAR,
                    recv_buf.data(), recv_counts.data(), displs.data(),
                    MPI_CHAR, 0, MPI_COMM_WORLD);

        const fs::path out_path = fs::path("output/log/integrals_all.txt");
        fs::create_directories(out_path.parent_path());
        std::ofstream ofs(out_path);
        if (!ofs)
        {
            std::cerr << "Failed to open " << out_path << " for writing.\n";
            return false;
        }
        // Write concatenated rank blocks in rank order.
        for (int r = 0; r < world_size; ++r)
        {
            ofs.write(recv_buf.data() + displs[r], recv_counts[r]);
            ofs << "\n";
        }
        ofs << std::flush;
        std::cout << "Integral dump written to: " << out_path << std::endl;
    }
    else
    {
        MPI_Gatherv(local_str.data(), local_len, MPI_CHAR,
                    nullptr, nullptr, nullptr, MPI_CHAR,
                    0, MPI_COMM_WORLD);
    }
#else
    // Serial: write directly.
    const fs::path out_path = fs::path("output/log/integrals_all.txt");
    fs::create_directories(out_path.parent_path());
    std::ofstream ofs(out_path);
    if (!ofs)
    {
        std::cerr << "Failed to open " << out_path << " for writing.\n";
        return false;
    }
    ofs << oss.str();
    std::cout << "Integral dump written to: " << out_path << std::endl;
#endif

    return true;
}

bool DumpCoefficients(const std::vector<std::vector<std::vector<mfem::DenseMatrix>>> &coeff,
                      const pbte::AngleQuadrature &quad,
                      bool is_root)
{
#ifdef MFEM_USE_MPI
    if (!is_root) { return true; }
#endif
    const fs::path out_dir = fs::path("output/log");
    fs::create_directories(out_dir);
    const fs::path fname = out_dir / "coeff_all.txt";
    std::ofstream ofs(fname);
    if (!ofs)
    {
        std::cerr << "Failed to open " << fname << " for writing.\n";
        return false;
    }

    const auto &dirs = quad.Directions();
    const int ndir = static_cast<int>(coeff.size());
    for (int d = 0; d < ndir; ++d)
    {
        const int nbranch = static_cast<int>(coeff[d].size());
        const auto &dir = dirs[d];
        for (int b = 0; b < nbranch; ++b)
        {
            const int nspec = static_cast<int>(coeff[d][b].size());
            for (int s = 0; s < nspec; ++s)
            {
                const auto &m = coeff[d][b][s];
                const int ndof = m.Height();
                const int ne = m.Width();
                ofs << "# dir " << d << " branch " << b << " spec " << s << "\n";
                ofs << "# ndof " << ndof << " ne " << ne << "\n";
                ofs << "# direction: ";
                for (size_t k = 0; k < dir.direction.size(); ++k)
                {
                    ofs << dir.direction[k];
                    if (k + 1 < dir.direction.size()) { ofs << " "; }
                }
                ofs << " weight " << dir.weight << "\n";

                for (int e = 0; e < ne; ++e)
                {
                    ofs << "elem " << e << "\n";
                    for (int i = 0; i < ndof; ++i)
                    {
                        ofs << m(i, e);
                        if (i + 1 < ndof) { ofs << " "; }
                    }
                    ofs << "\n";
                }
                ofs << "\n";
            }
        }
    }
    ofs << std::flush;
    std::cout << "Coefficient blocks written to: " << fname << std::endl;
    return true;
}

bool DumpTemperature(const mfem::DenseMatrix &Tc, bool is_root)
{
#ifdef MFEM_USE_MPI
    if (!is_root) { return true; }
#endif
    const fs::path out_dir = fs::path("output/log");
    fs::create_directories(out_dir);
    const fs::path fname = out_dir / "Tc_all.txt";
    std::ofstream ofs(fname);
    if (!ofs)
    {
        std::cerr << "Failed to open " << fname << " for writing.\n";
        return false;
    }

    const int ndof = Tc.Height();
    const int ne = Tc.Width();
    ofs << "# Tc matrix\n";
    ofs << "# ndof " << ndof << " ne " << ne << "\n";

    for (int e = 0; e < ne; ++e)
    {
        ofs << "elem " << e << "\n";
        for (int i = 0; i < ndof; ++i)
        {
            ofs << Tc(i, e);
            if (i + 1 < ndof) { ofs << " "; }
        }
        ofs << "\n";
    }

    ofs << std::flush;
    std::cout << "Macroscopic temperature Tc written to: " << fname << std::endl;
    return true;
}

mfem::Vector ComputeFaceNormal(const mfem::Mesh &mesh, int face_id)
{
    mfem::Array<int> verts;
    mesh.GetFaceVertices(face_id, verts);
    const int sdim = mesh.SpaceDimension();
    mfem::Vector n(sdim);
    n = 0.0;

    if (sdim == 2 && verts.Size() >= 2)
    {
        const double *v0 = mesh.GetVertex(verts[0]);
        const double *v1 = mesh.GetVertex(verts[1]);
        const double dx = v1[0] - v0[0];
        const double dy = v1[1] - v0[1];
        n[0] = dy;
        n[1] = -dx;
    }
    else if (sdim == 3 && verts.Size() >= 3)
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
        mfem::Vector c(3);
        c[0] = e1[1] * e2[2] - e1[2] * e2[1];
        c[1] = e1[2] * e2[0] - e1[0] * e2[2];
        c[2] = e1[0] * e2[1] - e1[1] * e2[0];
        n = c;
    }

    const double norm = n.Norml2();
    if (norm > 0.0)
    {
        n /= norm;
    }
    return n;
}

mfem::Vector ComputeOutwardFaceNormal(const mfem::Mesh &mesh,
                                      int face_id,
                                      int elem_id)
{
    mfem::Vector n = ComputeFaceNormal(mesh, face_id);
    const int sdim = mesh.SpaceDimension();

    mfem::Vector elem_c(sdim), face_c(sdim);
    elem_c = 0.0;
    face_c = 0.0;

    mfem::Array<int> e_verts;
    mesh.GetElementVertices(elem_id, e_verts);
    for (int v : e_verts)
    {
        const double *pv = mesh.GetVertex(v);
        for (int d = 0; d < sdim; ++d)
        {
            elem_c[d] += pv[d];
        }
    }
    if (e_verts.Size() > 0)
    {
        elem_c /= static_cast<double>(e_verts.Size());
    }

    mfem::Array<int> f_verts;
    mesh.GetFaceVertices(face_id, f_verts);
    for (int v : f_verts)
    {
        const double *pv = mesh.GetVertex(v);
        for (int d = 0; d < sdim; ++d)
        {
            face_c[d] += pv[d];
        }
    }
    if (f_verts.Size() > 0)
    {
        face_c /= static_cast<double>(f_verts.Size());
    }

    mfem::Vector to_face = face_c;
    to_face -= elem_c;
    if (n * to_face < 0.0)
    {
        n *= -1.0;
    }
    return n;
}
} // namespace utils
