#include "PhononModel/NonGraySMRT.hpp"
#include "Eigen/Dense"
#include "omp.h"
#include "GlobalConfig/GlobalConfig.hpp"
#include "SolidAngle/SolidAngle.hpp"
#include "SpatialMesh/SpatialMesh.hpp"
#include "PolyFem/BasisFunctions.hpp"
#include "PolyFem/PolyIntegral.hpp"
#include <iostream>

namespace PhononModel {
    template<int SpatialDim>
    NonGraySMRT_MPI<SpatialDim>::NonGraySMRT_MPI(const SpatialMesh::MeshPartitionInfo<SpatialDim>& partition_info, MPI_Comm communicator) : comm(communicator) {
        W.resize(CC.POLAR_DIM);
        K.resize(CC.POLAR_DIM);
        invKn.resize(CC.POLAR_DIM);
        Vg.resize(CC.POLAR_DIM);
        D.resize(CC.POLAR_DIM);
        HeatCap.resize(CC.POLAR_DIM);
        for(int i = 0; i < CC.POLAR_DIM; i++){
            W[i].resize(2, CC.NSPEC);
            K[i].resize(CC.NSPEC);
            invKn[i].resize(CC.NSPEC);
            Vg[i].resize(CC.NSPEC);
            D[i].resize(CC.NSPEC);
            HeatCap[i].resize(CC.NSPEC);
        }

        Eigen::VectorXd kb = Eigen::VectorXd::Zero(CC.NSPEC);
        for(int i = 1; i <= CC.NSPEC; i++){
            kb(i-1) = (2.0 * i - 1.0) / (2.0 * CC.NSPEC) * PC.K_RANGE[1];
        }

        // LA
        K[0] = kb;
        W[0].row(0) = PC.C_LA[0] * K[0].array() + PC.C_LA[1] * K[0].array().pow(2);
        Vg[0] = PC.C_LA[0] + 2.0 * PC.C_LA[1] * K[0].array();
        W[0].row(1) = PC.K_RANGE[1] * Vg[0].transpose().array();
        // Kn_inv[0] =  physics_c.L_REF / Vg[0].transpose().array() * (phonon_c.Ai * w[0].row(0).array().pow(4) + phonon_c.BL * pow(physics_c.T_REF, 3) * w[0].row(0).array().pow(2));
        invKn[0] = PC.Ai * W[0].row(0).array().pow(4) + PC.BL * pow(CC.T_REF, 3) * W[0].row(0).array().pow(2);
        // TA
        K[1] = kb;
        W[1].row(0) = PC.C_TA[0] * K[1].array() + PC.C_TA[1] * K[1].array().pow(2);
        Vg[1] = PC.C_TA[0] + 2.0 * PC.C_TA[1] * K[1].array();
        for(int i = 0; i < CC.NSPEC; i++){
            invKn[1](i) = PC.Ai * pow(W[1](0, i), 4);
            if(kb(i) < PC.K_RANGE[1] / 2.0){
                invKn[1](i) += PC.BT * W[1](0, i) * pow(CC.T_REF, 4);
            }
            else{
                invKn[1](i) += PC.BU * pow(W[1](0, i), 2) / sinh(CC.H * W[1](0, i) / CC.KB / CC.T_REF);
            }
            // Kn_inv[1](i) *= physics_c.L_REF / Vg[1](i);
        }
        W[1].row(1) = PC.K_RANGE[1] * Vg[1].transpose().array();

        D[0] = K[0].array().pow(2) / Vg[0].array() / 2.0 / pow(M_PI, 2);
        D[1] = K[1].array().pow(2) / Vg[1].array() / 2.0 / pow(M_PI, 2);

        HeatCapV = 0.0;
        for(int p = 0; p < CC.POLAR_DIM; p++){
            for(int j = 0; j < CC.NSPEC; j++){
                HeatCap[p](j) = pow(CC.H, 2) * pow(W[p](0, j), 2) * D[p](j) * exp(CC.H * W[p](0, j) / CC.KB / CC.T_REF) \
                                / pow(exp(CC.H * W[p](0, j) / CC.KB / CC.T_REF) - 1.0, 2) / CC.KB / pow(CC.T_REF, 2);
                HeatCapV += HeatCap[p](j) * invKn[p](j) * W[p](1, j);
            }
        }

        omp_set_num_threads(omp_get_max_threads());                        
        MPI_Comm_rank(comm, &world_rank);
        MPI_Comm_size(comm, &world_size);

        n_local_cells = partition_info.getPartitionCells(world_rank).size();
        local_ApproxCoeff_Vec.resize(n_local_cells * CC.POLAR_DIM * CC.NSPEC * CC.NPOLE * CC.NAZIM * CC.DOF);
        local_Tc_Vec.resize(n_local_cells * CC.DOF);
        local_Tv_Vec.resize(n_local_cells);
        local_Qc_Vec.resize(SpatialDim * n_local_cells * CC.DOF);
        local_Qv_Vec.resize(n_local_cells);
        local_Uc_Vec.resize(SpatialDim * n_local_cells * CC.DOF);

        if(world_rank == 0){
            global_ApproxCoeff.resize(CC.N_MESH_CELL, std::vector<std::vector<std::vector<std::vector<Eigen::VectorXd>>>> \
                                    (CC.POLAR_DIM, std::vector<std::vector<std::vector<Eigen::VectorXd>>> \
                                    (CC.NSPEC, std::vector<std::vector<Eigen::VectorXd>> \
                                    (CC.NPOLE, std::vector<Eigen::VectorXd> \
                                    (CC.NAZIM, Eigen::VectorXd::Zero(CC.DOF))))));

            global_Tc = Eigen::MatrixXd::Zero(CC.DOF, CC.N_MESH_CELL);
            global_Tv = Eigen::VectorXd::Zero(CC.N_MESH_CELL);
            global_Qc = std::vector<Eigen::MatrixXd>(CC.N_MESH_CELL, Eigen::MatrixXd::Zero(CC.DOF, CC.SPATIAL_DIM));
            global_Qv = std::vector<Eigen::VectorXd>(CC.N_MESH_CELL, Eigen::VectorXd::Zero(CC.SPATIAL_DIM));
            global_Uc = std::vector<Eigen::MatrixXd>(CC.N_MESH_CELL, Eigen::MatrixXd::Zero(CC.DOF, CC.SPATIAL_DIM));
        }

    }

    template<int SpatialDim>
    void NonGraySMRT_MPI<SpatialDim>::updateApproxCoeff(int idx_cell, int idx_polar, int idx_spec, int idx_pole, int idx_azim, const Eigen::VectorXd& Solution){
        global_ApproxCoeff[idx_cell][idx_polar][idx_spec][idx_pole][idx_azim] = Solution;
    }

    template<int SpatialDim>
    void NonGraySMRT_MPI<SpatialDim>::update_local_ApproxCoeff(int idx_cell, int idx_polar, int idx_spec, int idx_pole, int idx_azim, const Eigen::VectorXd& Solution, const SpatialMesh::MeshPartitionInfo<SpatialDim>& partition_info){
        int local_cell_idx = partition_info.getLocalCellIdx(world_rank).at(idx_cell);
        assert(local_cell_idx >= 0 && local_cell_idx < n_local_cells);
        int offset = local_cell_idx * CC.POLAR_DIM * CC.NSPEC * CC.NPOLE * CC.NAZIM * CC.DOF
                    + idx_polar * CC.NSPEC * CC.NPOLE * CC.NAZIM * CC.DOF
                    + idx_spec * CC.NPOLE * CC.NAZIM * CC.DOF
                    + idx_pole * CC.NAZIM * CC.DOF
                    + idx_azim * CC.DOF;
        std::copy(Solution.data(), Solution.data() + CC.DOF, local_ApproxCoeff_Vec.begin() + offset);
        // Eigen::Map<Eigen::VectorXd>(&local_ApproxCoeff_Vec[offset], CC.DOF) = Solution;
    }

    template<int SpatialDim>
    double NonGraySMRT_MPI<SpatialDim>::cal_residual_MPI(const Eigen::MatrixXd& IntMat){
        Eigen::VectorXd curTv = global_Tv;
        global_Tv = (global_Tc.transpose().array() * IntMat.array()).rowwise().sum();
        assert(global_Tv.norm() > 0.0);
        return (global_Tv - curTv).norm() / global_Tv.norm();
    }

    template<int SpatialDim>
    void NonGraySMRT_MPI<SpatialDim>::update_global_Tc(int cell_idx, const Eigen::VectorXd& cell_Tc){
        global_Tc.col(cell_idx) = cell_Tc;
    }

    template<int SpatialDim>
    void NonGraySMRT_MPI<SpatialDim>::output_3D_2Dslice_T_Q(std::vector<std::shared_ptr<SpatialMesh::Cell<3, 4>>>& Cells,
        SolidAngle& dsa, std::string results_dir, int fix_crd_idx, double crd)
    {
        // Determine physical-domain bounding box from mesh cells
        std::vector<std::vector<double>> Domain = {{std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()},
                                                {std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()}};
        for (const auto& cell : Cells) {
            const auto& verts = cell->getVertices();
            for (const auto& v : verts) {
                const auto& crd = v->getCoordinates();
                Domain[0][0] = std::min(Domain[0][0], crd(0));
                Domain[0][1] = std::max(Domain[0][1], crd(0));
                Domain[1][0] = std::min(Domain[1][0], crd(1));
                Domain[1][1] = std::max(Domain[1][1], crd(1));
            }
        }
        std::vector<int> N = {100, 100};

        std::vector<std::vector<int>> CellID(N[0], std::vector<int>(N[1], -1));
        Eigen::MatrixXd P(3, N[0]*N[1]);
        Eigen::MatrixXd refP(3, N[0]*N[1]);
        omp_set_num_threads(omp_get_max_threads());
        const double boundary_tol = 1e-12; // Small offset to avoid exact boundary points
        if(fix_crd_idx == 0){
            #pragma omp parallel for collapse(2)
            for(int i = 0; i < N[0]; i++){
                for(int j = 0; j < N[1]; j++){
                    P(0, j*N[0] + i) = crd;
                    P(1, j*N[0] + i) = double(j) / double(N[1] - 1) * (Domain[1][1] - Domain[1][0]) + Domain[1][0];
                    P(2, j*N[0] + i) = double(i) / double(N[0] - 1) * (Domain[0][1] - Domain[0][0]) + Domain[0][0];
                    // Clamp to slightly inside the domain
                    if (P(1, j*N[0] + i) >= Domain[1][1]) P(1, j*N[0] + i) = Domain[1][1] - boundary_tol;
                    if (P(2, j*N[0] + i) >= Domain[0][1]) P(2, j*N[0] + i) = Domain[0][1] - boundary_tol;
                }
            }
        }
        else if(fix_crd_idx == 1){
            #pragma omp parallel for collapse(2)
            for(int i = 0; i < N[0]; i++){
                for(int j = 0; j < N[1]; j++){
                    P(0, j*N[0] + i) = double(i) / double(N[0] - 1) * (Domain[0][1] - Domain[0][0]) + Domain[0][0];
                    P(1, j*N[0] + i) = crd;
                    P(2, j*N[0] + i) = double(j) / double(N[1] - 1) * (Domain[1][1] - Domain[1][0]) + Domain[1][0];
                    // Clamp to slightly inside the domain
                    if (P(0, j*N[0] + i) >= Domain[0][1]) P(0, j*N[0] + i) = Domain[0][1] - boundary_tol;
                    if (P(2, j*N[0] + i) >= Domain[1][1]) P(2, j*N[0] + i) = Domain[1][1] - boundary_tol;
                }
            }
        }
        else if(fix_crd_idx == 2){
            #pragma omp parallel for collapse(2)
            for(int i = 0; i < N[0]; i++){
                for(int j = 0; j < N[1]; j++){
                    P(0, j*N[0] + i) = double(i) / double(N[0] - 1) * (Domain[0][1] - Domain[0][0]) + Domain[0][0];
                    P(1, j*N[0] + i) = double(j) / double(N[1] - 1) * (Domain[1][1] - Domain[1][0]) + Domain[1][0];
                    P(2, j*N[0] + i) = crd;
                    // Clamp to slightly inside the domain
                    if (P(0, j*N[0] + i) >= Domain[0][1]) P(0, j*N[0] + i) = Domain[0][1] - boundary_tol;
                    if (P(1, j*N[0] + i) >= Domain[1][1]) P(1, j*N[0] + i) = Domain[1][1] - boundary_tol;
                }
            }
        }

        #pragma omp parallel for collapse(2)
        for(int i = 0; i < N[0]; i++){
            for(int j = 0; j < N[1]; j++){
                for(int k = 0; k < CC.N_MESH_CELL; k++){
                    if(Cells[k]->isPointInside(std::make_shared<SpatialMesh::Node<3>>(
                                                            P(0, j*N[0] + i), 
                                                            P(1, j*N[0] + i), 
                                                            P(2, j*N[0] + i))))
                    {
                        CellID[i][j] = k;
                        Eigen::MatrixXd Vertices = Cells[k]->getVerticesCoordinates();
                        refP.col(j*N[0] + i) = PolyFem::CordTrans(Vertices, P.col(j*N[0] + i));
                        break;
                    }
                }
                if(CellID[i][j] == -1){
                    throw std::runtime_error("Point not found in any cell");
                }
            }
        }

        std::vector<std::vector<double>> T(N[0], std::vector<double>(N[1], 0.0));
        std::vector<std::vector<double>> Qx(N[0], std::vector<double>(N[1], 0.0));
        std::vector<std::vector<double>> Qy(N[0], std::vector<double>(N[1], 0.0));
        std::vector<std::vector<double>> Qz(N[0], std::vector<double>(N[1], 0.0));

        std::vector<Polynomial::Polynomial> RefBasis = PolyFem::ReferenceBasis(3, CC.POLYDEG);
        Eigen::MatrixXd Evl(CC.DOF, N[0]*N[1]);
        for(int k = 0; k < RefBasis.size(); k++){
            Evl.row(k) = RefBasis[k].evaluateBatch(refP).transpose();
        }

        #pragma omp parallel for collapse(2) schedule(static)
        for(int i = 0; i < N[0]; i++){
            for(int j = 0; j < N[1]; j++){
                if(CellID[i][j] == -1){
                    std::cout << "Point " << i << " " << j << " (" << P(0, j*N[0] + i) << ", " << P(1, j*N[0] + i) << ") not found in any cell." << std::endl;
                    // throw std::runtime_error("Point not found in any cell");
                }
                double local_s = 0.0;
                double local_sx = 0.0;
                double local_sy = 0.0;
                double local_sz = 0.0;
                const int point_index = j*N[0] + i;
                const int cell_index = CellID[i][j];
                const auto eval_col = Evl.col(point_index);
                for(int p = 0; p < CC.POLAR_DIM; p++){
                    for(int k = 0; k < CC.NSPEC; k++){
                        const double Kn_inv = invKn[p](k);
                        const double vg = Vg[p](k);
                        const double dW = W[p](1, k);
                        for(int ip = 0; ip < CC.NPOLE; ip++){
                            for(int ja = 0; ja < CC.NAZIM; ja++){
                                const double wt = dsa.wt()[ip][ja];
                                const double a = Kn_inv * global_ApproxCoeff[cell_index][p][k][ip][ja].dot(eval_col) * wt * dW;
                                local_s += a;
                                const auto dir = dsa.dir()[ip][ja];
                                local_sx += vg * a * dir(0);
                                local_sy += vg * a * dir(1);
                                local_sz += vg * a * dir(2);
                            }
                        }
                    }
                }
                T[i][j] = local_s / HeatCapV;
                Qx[i][j] = local_sx;
                Qy[i][j] = local_sy;
                Qz[i][j] = local_sz;
            }
        }

        std::ofstream write_output(results_dir);
        assert(write_output.is_open());
        write_output.setf(std::ios::fixed);
        write_output.precision(16);
        write_output << "x" << " " << "y" << " " << "z" << " " << "T" << " " << "Qx" << " " << "Qy" << " " << "Qz" << std::endl;
        for(int iy = 0; iy < N[1]; iy++){
            for(int ix = 0; ix < N[0]; ix++){
                write_output << P(0, iy*N[0] + ix) << " " << P(1, iy*N[0] + ix) << " " << P(2, iy*N[0] + ix) << " " << T[ix][iy] << " " << Qx[ix][iy] << " " << Qy[ix][iy] << " " << Qz[ix][iy] << std::endl;
            }
        }
        
        write_output.close();
    }
}

template class PhononModel::NonGraySMRT_MPI<2>;
template class PhononModel::NonGraySMRT_MPI<3>;