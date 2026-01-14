#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
#include <omp.h>

#include "Eigen/Dense"
#include "GlobalConfig/GlobalConfig.hpp"
#include "SpatialMesh/SpatialMesh.hpp"
#include "SolidAngle/SolidAngle.hpp"
#include "PolyFem/PolyIntegral.hpp"
#include "DGSolver/PBTE_NonGraySMRT.hpp"
#include "PhononModel/NonGraySMRT.hpp"

using namespace Eigen;
using namespace std;
using namespace Polynomial;
using namespace PhononModel;
using namespace DGSolver;

ControlConstants CC;
PhononConstants PC;

int main(int argc, char* argv[]){
    cout.precision(12);
    
    // Print usage information
    if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        cout << "Usage: " << argv[0] << " [num_partitions]" << endl;
        cout << "  num_partitions: Number of mesh partitions (default: 4)" << endl;
        cout << "  -h, --help:     Show this help message" << endl;
        return 0;
    }

    // Initialize MPI early so we know rank/size before partitioning
    MPI_Init(&argc, &argv);
    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    PC.LoadFromFile("config/control/Si_PhononModel.yaml");
    CC.LoadFromFile("config/control/Control.yaml");
    cout << ">>> Parameter loading completed successfully!" << endl;

    SolidAngle DSA(CC.MATERIAL_DIM, CC.NPOLE, CC.NAZIM, CC.SOLID_ANGLE_PATTERN);

    SpatialMesh::SpatialMesh<DIM> mesh(CC.MESH_FILE, DSA);
    auto Cells = mesh.getCells();
    auto Faces = mesh.getFaces();
    auto CompOrder = mesh.getComputationOrder();
    auto Outflow = mesh.getOutflow();

    CC.N_MESH_CELL = Cells.size();
    mesh.SaveMeshInfo("output/" + std::to_string(DIM) + "D/log/mesh_info.txt");

    // Mesh partitioning for parallel computation
    #ifdef HAVE_METIS
    cout << ">>> Starting mesh partitioning..." << endl;
    int num_partitions = 4;  // Default number of partitions (will be reconciled with MPI size)
    if (argc > 1) { num_partitions = std::stoi(argv[1]); }
    // Reconcile partitions with MPI ranks:
    // - If running single-rank, force single partition and disable MPI solver path.
    // - If running multi-rank, match partitions to ranks to avoid invalid MPI destinations.
    if (world_size == 1) {
        if (num_partitions != 1 && world_rank == 0) {
            cout << ">>> Single MPI rank detected; overriding num_partitions " << num_partitions << " -> 1" << endl;
        }
        num_partitions = 1;
    } else if (num_partitions != world_size) {
        if (world_rank == 0) {
            cout << ">>> Warning: num_partitions (" << num_partitions << ") != MPI ranks ("
                 << world_size << "). Overriding partitions to " << world_size << endl;
        }
        num_partitions = world_size;
    }
    
    bool partition_success = mesh.partitionMesh(num_partitions);
    const auto* partition_info = mesh.getPartitionInfo();
    
    if (partition_success) {
        cout << ">>> Mesh partitioned successfully into " << num_partitions << " parts" << endl;
        
        // Save partition information
        if (partition_info) {
            // Save partition mapping to file
            std::ofstream partition_file("output/" + std::to_string(DIM) + "D/log/partition_info.txt");
            if (partition_file.is_open()) {
                partition_file << "# Partition information for " << num_partitions << " partitions" << endl;
                partition_file << "# Format: cell_id partition_id" << endl;
                
                const auto& cell_partition = partition_info->getCellPartition();
                for (size_t i = 0; i < cell_partition.size(); ++i) {
                    partition_file << i << " " << cell_partition[i] << endl;
                }
                partition_file.close();
                cout << ">>> Partition info saved to output/" << DIM << "D/log/partition_info.txt" << endl;
            }
            
            // Print partition statistics
            // partition_info->printPartitionStatistics();
        }
    } else {
        cout << ">>> Warning: Mesh partitioning failed, continuing with single partition" << endl;
    }

    if (mesh.isPartitioned()) {
        cout << ">>> Final partition summary:" << endl;
        if (partition_info) {
            const auto& partition_cells = partition_info->getPartitionCells();
            cout << "  - Total partitions: " << partition_info->getNumPartitions() << endl;
            cout << "  - Cells per partition: ";
            for (int i = 0; i < partition_info->getNumPartitions(); ++i) {
                cout << partition_cells[i].size();
                if (i < partition_info->getNumPartitions() - 1) cout << ", ";
            }
            cout << endl;
            
            // Calculate load balance
            auto min_it = std::min_element(partition_cells.begin(), partition_cells.end(),
                [](const std::vector<int>& a, const std::vector<int>& b) {
                    return a.size() < b.size();
                });
            auto max_it = std::max_element(partition_cells.begin(), partition_cells.end(),
                [](const std::vector<int>& a, const std::vector<int>& b) {
                    return a.size() < b.size();
                });
            int min_cells = min_it->size();
            int max_cells = max_it->size();
            double balance_ratio = static_cast<double>(min_cells) / max_cells;
            cout << "  - Load balance ratio: " << balance_ratio << " (1.0 = perfect balance)" << endl;
            cout << "================================= " << endl;
        }
    }
    #else
    cout << ">>> METIS not available, running without partitioning" << endl;
    #endif

    PolyFem::Integral<DIM> Int(mesh);
    // Int.save("output/" + std::to_string(DIM) + "D/log");
    Eigen::MatrixXd IntMat = Int.getIntMat();
    std::vector<Eigen::MatrixXd> MassMat = Int.getMassMat();
    std::vector<std::vector<Eigen::MatrixXd>> StfMat = Int.getStiffMat();
    std::vector<std::vector<Eigen::MatrixXd>> MassFaceMat = Int.getMassFaceMat();
    std::vector<std::vector<Eigen::MatrixXd>> FluxMat = Int.getFluxMat();
    std::vector<std::vector<Eigen::VectorXd>> IntFaceMat = Int.getIntFaceMat();

    if (world_size == 1) {
        // Run serial (non-MPI) path when only one rank is present
        NonGraySMRT<DIM> non_gray_smrt;
        PBTE_NonGraySMRT<DIM> solver(MassMat, StfMat, MassFaceMat, non_gray_smrt, CompOrder, DSA, Outflow);
        solver.solve(Cells, IntMat, MassMat, MassFaceMat, FluxMat, non_gray_smrt, CompOrder, DSA, Outflow);
        non_gray_smrt.output_3D_2Dslice_T_Q(Cells, DSA, CC.output_path, 2, 0.4*CC.L_REF);
    } else {
        // Safety: ensure partition info exists and matches MPI size
        if (!partition_info || partition_info->getNumPartitions() != world_size) {
            if (world_rank == 0) {
                std::cerr << "Error: Partition info invalid or does not match MPI ranks." << std::endl;
            }
            MPI_Finalize();
            return 1;
        }
        NonGraySMRT_MPI<DIM> non_gray_smrt_mpi(*partition_info, MPI_COMM_WORLD);
        PBTE_NonGraySMRT_MPI<DIM> solver_mpi(Cells, *partition_info, non_gray_smrt_mpi, MPI_COMM_WORLD);
        solver_mpi.solve(Cells, *partition_info, IntMat, MassMat, MassFaceMat, StfMat, FluxMat, non_gray_smrt_mpi, DSA, Outflow);
        if (world_rank == 0) {
            non_gray_smrt_mpi.output_3D_2Dslice_T_Q(Cells, DSA, CC.output_path, 2, 0.4*CC.L_REF);
        }
    }

    MPI_Finalize();
    
    return 0;
}