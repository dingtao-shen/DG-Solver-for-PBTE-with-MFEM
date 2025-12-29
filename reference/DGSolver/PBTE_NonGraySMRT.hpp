#pragma once

#include "DGSolver/DGSolver.hpp"
#include "SpatialMesh/SpatialMesh.hpp"
#include "SolidAngle/SolidAngle.hpp"
#include "Eigen/Dense"
#include "GlobalConfig/GlobalConfig.hpp"
#include "PolyFem/PolyIntegral.hpp"
#include "PhononModel/NonGraySMRT.hpp"
#include "SpatialMesh/MeshPartitioning.hpp"
#include <mpi.h>

namespace DGSolver {

    template<int dim>
    class PBTE_NonGraySMRT : public DGSolver<dim> {
        private:
            std::vector<std::vector<std::vector<std::vector<std::vector<Eigen::PartialPivLU<Eigen::MatrixXd>>>>>> PreComputedLU;
            double dt_inv;
        public:
            PBTE_NonGraySMRT(std::vector<Eigen::MatrixXd>& MassMat, 
                        std::vector<std::vector<Eigen::MatrixXd>>& StfMat, 
                        std::vector<std::vector<Eigen::MatrixXd>>& MassOnFace,
                        PhononModel::NonGraySMRT<dim>& non_gray_smrt,
                        std::vector<std::vector<std::vector<int>>>& comp_odr,
                        SolidAngle& dsa,
                        std::vector<std::vector<Eigen::MatrixXd>>& outflow);
            void solve(std::vector<std::shared_ptr<SpatialMesh::Cell<dim, dim+1>>>& Cells, 
                       Eigen::MatrixXd& IntMat,
                       std::vector<Eigen::MatrixXd>& MassMat, 
                       std::vector<std::vector<Eigen::MatrixXd>>& MassOnFace, 
                       std::vector<std::vector<Eigen::MatrixXd>>& FluxInt,
                       PhononModel::NonGraySMRT<dim>& non_gray_smrt,
                       std::vector<std::vector<std::vector<int>>>& comp_odr,
                       SolidAngle& dsa,
                       std::vector<std::vector<Eigen::MatrixXd>>& outflow);
    };

    template<int dim>
    class PBTE_NonGraySMRT_MPI : public DGSolver<dim> {
        private:
            using LUDecomp = Eigen::PartialPivLU<Eigen::MatrixXd>;
            std::unordered_map<int, std::vector<std::vector<Eigen::VectorXd>>> local_source;
            std::vector<double> local_solutions;
            std::vector<double> nbr_solutions;
            std::vector<double> local_Tc;

            double dt_inv;
            double residual;
            int iter_step;
            int single_cell_msg_size;

            int world_rank;
            int world_size;
            MPI_Comm comm;

            // Reusable communication buffers to avoid per-iteration allocations
            // recv_buffers: key = source rank
            std::unordered_map<int, std::vector<double>> recv_buffers;
            // send_buffers: key = destination rank
            std::unordered_map<int, std::vector<double>> send_buffers;

            // Precomputed LU for local cells to avoid per-iteration factorization:
            // Dimensions: [POLAR_DIM][NSPEC][NPOLE][NAZIM][n_local_cells]
            std::vector<std::vector<std::vector<std::vector<std::vector<LUDecomp>>>>> PreComputedLU_local;
            bool precomputed_lu_ready = false;

        public:
            PBTE_NonGraySMRT_MPI(std::vector<std::shared_ptr<SpatialMesh::Cell<dim, dim+1>>>& Cells,
                                const SpatialMesh::MeshPartitionInfo<dim>& partition_info,
                                const PhononModel::NonGraySMRT_MPI<dim>& non_gray_smrt_mpi,
                                MPI_Comm communicator = MPI_COMM_WORLD);
            ~PBTE_NonGraySMRT_MPI();

            void exchange_solutions(const SpatialMesh::MeshPartitionInfo<dim>& partition_info);
            void gather_solutions(const SpatialMesh::MeshPartitionInfo<dim>& partition_info, PhononModel::NonGraySMRT_MPI<dim>& non_gray_smrt);
            void update_Tc(const SpatialMesh::MeshPartitionInfo<dim>& partition_info, PhononModel::NonGraySMRT_MPI<dim>& non_gray_smrt, Eigen::MatrixXd& IntMat);
            void solve(std::vector<std::shared_ptr<SpatialMesh::Cell<dim, dim+1>>>& Cells,
                const SpatialMesh::MeshPartitionInfo<dim>& partition_info,
                Eigen::MatrixXd& IntMat,
                std::vector<Eigen::MatrixXd>& MassMat, 
                std::vector<std::vector<Eigen::MatrixXd>>& MassOnFace, 
                std::vector<std::vector<Eigen::MatrixXd>>& StfMat,
                std::vector<std::vector<Eigen::MatrixXd>>& FluxInt,
                PhononModel::NonGraySMRT_MPI<dim>& non_gray_smrt,
                SolidAngle& dsa,
                std::vector<std::vector<Eigen::MatrixXd>>& outflow);
    };

} // namespace DGSolver