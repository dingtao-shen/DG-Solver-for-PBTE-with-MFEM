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
    NonGraySMRT<SpatialDim>::NonGraySMRT(){
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

        ApproxCoeff.resize(CC.POLAR_DIM, std::vector<std::vector<std::vector<Eigen::MatrixXd>>>(CC.NSPEC, 
                std::vector<std::vector<Eigen::MatrixXd>>(CC.NPOLE, std::vector<Eigen::MatrixXd>(CC.NAZIM, 
                Eigen::MatrixXd::Zero(CC.DOF, CC.N_MESH_CELL)))));
        for(int p = 0; p < CC.POLAR_DIM; p++){
            for(int s = 0; s < CC.NSPEC; s++){
                for(int i = 0; i < CC.NPOLE; i++){
                    for(int j = 0; j < CC.NAZIM; j++){
                        ApproxCoeff[p][s][i][j].setZero();
                    }
                }
            }
        }

        Tc = Eigen::MatrixXd::Zero(CC.DOF, CC.N_MESH_CELL);
        Tv = Eigen::VectorXd::Zero(CC.N_MESH_CELL);
        Qc = std::vector<Eigen::MatrixXd>(CC.SPATIAL_DIM, Eigen::MatrixXd::Zero(CC.DOF, CC.N_MESH_CELL));
        Qv = std::vector<Eigen::VectorXd>(CC.SPATIAL_DIM, Eigen::VectorXd::Zero(CC.N_MESH_CELL));
        Uc = std::vector<Eigen::MatrixXd>(CC.SPATIAL_DIM, Eigen::MatrixXd::Zero(CC.DOF, CC.N_MESH_CELL));

        std::cout << ">>> PhononModel_NonGray_SMRT initialized" << std::endl;
    }

    template<int SpatialDim>
    void NonGraySMRT<SpatialDim>::updateApproxCoeff(int idx_polar, int idx_spec, int idx_pole, int idx_azim, int idx_cell, const Eigen::VectorXd& Solution){
        ApproxCoeff[idx_polar][idx_spec][idx_pole][idx_azim].col(idx_cell) = Solution;
    }

    template<int SpatialDim>
    double NonGraySMRT<SpatialDim>::update(const SolidAngle& dsa, const Eigen::MatrixXd& IntMat){
        Eigen::VectorXd curTv = Tv;
        Tc.setZero();
        Tv.setZero();
        
        #pragma omp parallel for schedule(static)
        for(int l = 0; l < CC.N_MESH_CELL; l++){
            Eigen::VectorXd local_col = Eigen::VectorXd::Zero(CC.DOF);
            for(int p = 0; p < CC.POLAR_DIM; p++){
                for(int ik = 0; ik < CC.NSPEC; ik++){
                    for(int i = 0; i < CC.NPOLE; i++){
                        for(int j = 0; j < CC.NAZIM; j++){
                            local_col += invKn[p](ik) * dsa.wt()[i][j] * W[p](1, ik) * ApproxCoeff[p][ik][i][j].col(l);
                        }
                    }
                }
            }
            // Tc was zeroed before; assign the accumulated column directly
            Tc.col(l) = local_col;
        }

        Tc = 1.0 / HeatCapV * Tc.array();
        Tv = (Tc.transpose().array() * IntMat.array()).rowwise().sum();
        
        assert(Tv.norm() > 0.0);
        return (Tv - curTv).norm() / Tv.norm();
    }

    template<int SpatialDim>
    double NonGraySMRT<SpatialDim>::cal_residual_MPI(const Eigen::MatrixXd& IntMat){
        Eigen::VectorXd curTv = Tv;
        Tv = (Tc.transpose().array() * IntMat.array()).rowwise().sum();
        assert(Tv.norm() > 0.0);
        return (Tv - curTv).norm() / Tv.norm();
    }

    /***************************************2D*************************************************/

    template<int SpatialDim>
    void NonGraySMRT<SpatialDim>::output_2D_slice_T_Q(std::vector<std::shared_ptr<SpatialMesh::Cell<2, 3>>>& Cells, SolidAngle& dsa, std::string results_dir)
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
        
        // Debug: print domain bounds
        std::cout << "Domain bounds: x=[" << Domain[0][0] << ", " << Domain[0][1] 
                  << "], y=[" << Domain[1][0] << ", " << Domain[1][1] << "]" << std::endl;
        
        std::vector<int> N = {50, 50};
        std::vector<std::vector<int>> CellID(N[0], std::vector<int>(N[1], -1));
        Eigen::MatrixXd P(2, N[0]*N[1]);
        Eigen::MatrixXd refP(2, N[0]*N[1]);

        omp_set_num_threads(omp_get_max_threads());

        const double boundary_tol = 1e-12; // Small offset to avoid exact boundary points
        #pragma omp parallel for collapse(2)
        for(int i = 0; i < N[0]; i++){
            for(int j = 0; j < N[1]; j++){
                P(0, j*N[0] + i) = double(i) / double(N[0] - 1) * (Domain[0][1] - Domain[0][0]) + Domain[0][0];
                P(1, j*N[0] + i) = double(j) / double(N[1] - 1) * (Domain[1][1] - Domain[1][0]) + Domain[1][0];
                // Clamp to slightly inside the domain to avoid exact-boundary classification issues
                if (P(0, j*N[0] + i) >= Domain[0][1]) P(0, j*N[0] + i) = Domain[0][1] - boundary_tol;
                if (P(1, j*N[0] + i) >= Domain[1][1]) P(1, j*N[0] + i) = Domain[1][1] - boundary_tol;
            }
        }
        
        // Debug: print sampling range
        std::cout << "Sampling range: x=[" << P(0, 0) << ", " << P(0, N[0]*N[1]-1) 
                  << "], y=[" << P(1, 0) << ", " << P(1, N[0]*N[1]-1) << "]" << std::endl;

        #pragma omp parallel for collapse(2)
        for(int i = 0; i < N[0]; i++){
            for(int j = 0; j < N[1]; j++){
                for(int k = 0; k < CC.N_MESH_CELL; k++){
                    if(Cells[k]->isPointInside(std::make_shared<SpatialMesh::Node<2>>(
                                                        P(0, j*N[0] + i), 
                                                        P(1, j*N[0] + i))))
                    {
                        CellID[i][j] = k;
                        Eigen::MatrixXd Vertices = Cells[k]->getVerticesCoordinates();
                        refP.col(j*N[0] + i) = PolyFem::CordTrans(Vertices, P.col(j*N[0] + i));
                        break;
                    }
                }
            }
        }
        std::vector<std::vector<double>> T(N[0], std::vector<double>(N[1], 0.0));
        std::vector<std::vector<double>> Qx(N[0], std::vector<double>(N[1], 0.0));
        std::vector<std::vector<double>> Qy(N[0], std::vector<double>(N[1], 0.0));
        std::vector<Polynomial::Polynomial> RefBasis = PolyFem::ReferenceBasis(2, CC.POLYDEG);
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
                const int point_index = j*N[0] + i;
                const int cell_index = CellID[i][j];
                const auto eval_col = Evl.col(point_index);
                // for(int p = 0; p < CC.POLAR_DIM; p++){
                //     for(int k = 0; k < CC.NSPEC; k++){
                //         const double Kn_inv = invKn[p](k);
                //         const double vg = Vg[p](k);
                //         const double dW = W[p](1, k);
                //         for(int ip = 0; ip < CC.NPOLE; ip++){
                //             for(int ja = 0; ja < CC.NAZIM; ja++){
                //                 const double wt = dsa.wt()[ip][ja];
                //                 const auto dir = dsa.dir()[ip][ja];
                //                 const double a = Kn_inv * ApproxCoeff[p][k][ip][ja].col(cell_index).dot(eval_col) * wt * dW;
                //                 local_s += a;
                //                 local_sx += vg * a * dir(0);
                //                 local_sy += vg * a * dir(1);
                //             }
                //         }
                //     }
                // }
                // T[i][j] = local_s / HeatCapV;
                // Qx[i][j] = local_sx;
                // Qy[i][j] = local_sy;

                T[i][j] = Tc.col(cell_index).dot(eval_col);
            }
        }

        std::ofstream write_output(results_dir);
        assert(write_output.is_open());
        write_output.setf(std::ios::fixed);
        write_output.precision(16);
        write_output << "x" << " " << "y" << " " << "T" << " " << "Qx" << " " << "Qy" << std::endl;
        for(int i = 0; i < N[0]; i++){
            for(int j = 0; j < N[1]; j++){
                write_output << P(0, j*N[0] + i) << " " << P(1, j*N[0] + i) << " " << T[i][j] << " " << Qx[i][j] << " " << Qy[i][j] << std::endl;
            }
        }
        write_output.close();
    }

    /***************************************3D*************************************************/
    template<int SpatialDim>
    void NonGraySMRT<SpatialDim>::output_3D_1Dslice_T_Q(std::vector<std::shared_ptr<SpatialMesh::Cell<3, 4>>>& Cells,
            SolidAngle& dsa, std::string results_dir, int dir, double crd1, double crd2)
    {
        int N = 100;

        std::vector<int> CellID(N, -1);
        Eigen::MatrixXd P(3, N);
        Eigen::MatrixXd refP(3, N);
        omp_set_num_threads(omp_get_max_threads());
        const double boundary_tol = 1e-12; // Small offset to avoid exact boundary points
        if(dir == 0){
            for(int i = 0; i < N; i++){
                P(0, i) = double(i) / double(N - 1) * CC.L_REF;
                P(1, i) = crd1;
                P(2, i) = crd2;
                if(P(0, i) >= CC.L_REF) P(0, i) = CC.L_REF - boundary_tol;
                if(P(0, i) <= 0.0) P(0, i) = boundary_tol;
            }
        }
        else if(dir == 1){
            for(int i = 0; i < N; i++){
                P(0, i) = crd1;
                P(1, i) = double(i) / double(N - 1) * CC.L_REF;
                P(2, i) = crd2;
                if(P(1, i) >= CC.L_REF) P(1, i) = CC.L_REF - boundary_tol;
                if(P(1, i) <= 0.0) P(1, i) = boundary_tol;
            }
        }
        else if(dir == 2){
            for(int i = 0; i < N; i++){
                P(0, i) = crd1;
                P(1, i) = crd2;
                P(2, i) = double(i) / double(N - 1) * CC.L_REF;
                if(P(2, i) >= CC.L_REF) P(2, i) = CC.L_REF - boundary_tol;
                if(P(2, i) <= 0.0) P(2, i) = boundary_tol;
            }
        }
        else{
            throw std::runtime_error("Invalid direction");
        }

        for(int i = 0; i < N; i++){
            for(int k = 0; k < CC.N_MESH_CELL; k++){
                if(Cells[k]->isPointInside(std::make_shared<SpatialMesh::Node<3>>(
                                                        P(0, i), 
                                                        P(1, i), 
                                                        P(2, i))))
                {
                    CellID[i] = k;
                    Eigen::MatrixXd Vertices = Cells[k]->getVerticesCoordinates();
                    refP.col(i) = PolyFem::CordTrans(Vertices, P.col(i));
                    break;
                }
            }
            if(CellID[i] == -1){
                throw std::runtime_error("Point not found in any cell");
            }
            
        }

        std::vector<double> T(N, 0.0);
        std::vector<double> Qx(N, 0.0);
        std::vector<double> Qy(N, 0.0);
        std::vector<double> Qz(N, 0.0);

        std::vector<Polynomial::Polynomial> RefBasis = PolyFem::ReferenceBasis(3, CC.POLYDEG);
        Eigen::MatrixXd Evl(CC.DOF, N);
        for(int k = 0; k < RefBasis.size(); k++){
            Evl.row(k) = RefBasis[k].evaluateBatch(refP).transpose();
        }

        for(int i = 0; i < N; i++){
            if(CellID[i] == -1){
                std::cout << "Point " << i << " (" << P(0, i) << ", " << P(1, i) << ", " << P(2, i) << ") not found in any cell." << std::endl;
                // throw std::runtime_error("Point not found in any cell");
            }
            double local_s = 0.0;
            double local_sx = 0.0;
            double local_sy = 0.0;
            double local_sz = 0.0;
            const int point_index = i;
            const int cell_index = CellID[i];
            const auto eval_col = Evl.col(point_index);
            for(int p = 0; p < CC.POLAR_DIM; p++){
                for(int k = 0; k < CC.NSPEC; k++){
                    const double Kn_inv = invKn[p](k);
                    const double vg = Vg[p](k);
                    const double dW = W[p](1, k);
                    for(int ip = 0; ip < CC.NPOLE; ip++){
                        for(int ja = 0; ja < CC.NAZIM; ja++){
                            const double wt = dsa.wt()[ip][ja];
                            const double a = Kn_inv * ApproxCoeff[p][k][ip][ja].col(cell_index).dot(eval_col) * wt * dW;
                            local_s += a;
                            const auto dir = dsa.dir()[ip][ja];
                            local_sx += vg * a * dir(0);
                            local_sy += vg * a * dir(1);
                            local_sz += vg * a * dir(2);
                        }
                    }
                }
            }
            T[i] = local_s / HeatCapV;
            Qx[i] = local_sx;
            Qy[i] = local_sy;
            Qz[i] = local_sz;
            
        }

        std::ofstream write_output(results_dir);
        assert(write_output.is_open());
        write_output.setf(std::ios::fixed);
        write_output.precision(16);
        write_output << "x" << " " << "y" << " " << "z" << " " << "T" << " " << "Qx" << " " << "Qy" << " " << "Qz" << std::endl;
        for(int i = 0; i < N; i++){
            write_output << P(0, i) << " " << P(1, i) << " " << P(2, i) << " " << T[i] << " " << Qx[i] << " " << Qy[i] << " " << Qz[i] << std::endl;
        }
        
        write_output.close();
    }

    template<int SpatialDim>
    void NonGraySMRT<SpatialDim>::output_3D_2Dslice_T_Q(std::vector<std::shared_ptr<SpatialMesh::Cell<3, 4>>>& Cells,
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
                                const double a = Kn_inv * ApproxCoeff[p][k][ip][ja].col(cell_index).dot(eval_col) * wt * dW;
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

template class PhononModel::NonGraySMRT<2>;
template class PhononModel::NonGraySMRT<3>;
