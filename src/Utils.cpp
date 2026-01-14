#include "Utils.hpp"
#include "ElementIntegrator.hpp"

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
} // namespace utils
