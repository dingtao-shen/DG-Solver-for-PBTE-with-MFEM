#include "DGSolver/PBTE_NonGraySMRT.hpp"
#include <limits>
#include <cstring>
#include <stdexcept>
#include <algorithm>

namespace DGSolver {

    template<int dim>
    PBTE_NonGraySMRT<dim>::PBTE_NonGraySMRT(std::vector<Eigen::MatrixXd>& MassMat, 
                                      std::vector<std::vector<Eigen::MatrixXd>>& StfMat, 
                                      std::vector<std::vector<Eigen::MatrixXd>>& MassOnFace,
                                      PhononModel::NonGraySMRT<dim>& non_gray_smrt,
                                      std::vector<std::vector<std::vector<int>>>& comp_odr,
                                      SolidAngle& dsa,
                                      std::vector<std::vector<Eigen::MatrixXd>>& outflow)
    {
        PreComputedLU.resize(CC.POLAR_DIM, std::vector<std::vector<std::vector<std::vector<Eigen::PartialPivLU<Eigen::MatrixXd>>>>>(
                            CC.NSPEC, std::vector<std::vector<std::vector<Eigen::PartialPivLU<Eigen::MatrixXd>>>>(
                            CC.NPOLE, std::vector<std::vector<Eigen::PartialPivLU<Eigen::MatrixXd>>>(
                            CC.NAZIM, std::vector<Eigen::PartialPivLU<Eigen::MatrixXd>>(
                            CC.N_MESH_CELL, Eigen::PartialPivLU<Eigen::MatrixXd>(Eigen::MatrixXd::Zero(CC.DOF, CC.DOF)))))));

        dt_inv = 0.0;
        for(int p = 0; p < CC.POLAR_DIM; p++){
            dt_inv = std::max(non_gray_smrt.getInvKn(p).maxCoeff(), dt_inv);
        }

        omp_set_num_threads(omp_get_max_threads());
        for(int p = 0; p < CC.POLAR_DIM; p++){
            for(int s = 0; s < CC.NSPEC; s++){
                const auto dir = dsa.dir();
                const auto vg = non_gray_smrt.getVg(p, s);
                #pragma omp parallel for collapse(2) schedule(static)
                for(int j1 = 0; j1 < CC.NPOLE; j1++){
                    for(int j2 = 0; j2 < CC.NAZIM; j2++){
                        for(int l = 0; l < CC.N_MESH_CELL; l++){
                            int CellID = comp_odr[j1][j2][l];

                            Eigen::MatrixXd Coeff = dt_inv * MassMat[CellID].array();

                            for(int i = 0; i < dim; i++){
                                Coeff = Coeff.array() - vg * (dir[j1][j2](i) * StfMat[CellID][i].array());
                            }

                            for(int il = 0; il < dim+1; il++){
                                double f = outflow[j1][j2](il, CellID);
                                Coeff = Coeff.array() + 0.5 * vg * (f + abs(f)) * MassOnFace[CellID][il].array();
                            }

                            PreComputedLU[p][s][j1][j2][CellID] = Coeff.lu();
                        }
                    }
                }
            }
        }

        std::cout << "   >>> DGSolver_PBTE_NonGraySMRT initialized" << std::endl;
    }

    template<int dim>
    void PBTE_NonGraySMRT<dim>::solve(std::vector<std::shared_ptr<SpatialMesh::Cell<dim, dim+1>>>& Cells,
                                   Eigen::MatrixXd& IntMat,
                                   std::vector<Eigen::MatrixXd>& MassMat, 
                                   std::vector<std::vector<Eigen::MatrixXd>>& MassOnFace, 
                                   std::vector<std::vector<Eigen::MatrixXd>>& FluxInt,
                                   PhononModel::NonGraySMRT<dim>& non_gray_smrt,
                                   std::vector<std::vector<std::vector<int>>>& comp_odr,
                                   SolidAngle& dsa,
                                   std::vector<std::vector<Eigen::MatrixXd>>& outflow)
    {
        std::ofstream step_res_file("output/" + std::to_string(DIM) + "D/log/PBTE_NonGraySMRT_step_resisual.txt");
        if (!step_res_file.is_open())
        {
            throw std::runtime_error("Cannot open step_res.txt for writing");
        }
        int step = 1;
        double res = 1000000;
        // Precompute MassMat^T once to avoid repeated transposes each step
        std::vector<Eigen::MatrixXd> MassMatT(MassMat.size());
        for (size_t i = 0; i < MassMat.size(); ++i) {
            MassMatT[i] = MassMat[i].transpose();
        }
        while(res > CC.TOL && step < CC.TMAX){

            omp_set_num_threads(omp_get_max_threads());
            
            #pragma omp parallel
            {
                Eigen::VectorXd rhs;
                Eigen::VectorXd sol;
                rhs.resize(CC.DOF);
                sol.resize(CC.DOF);
                #pragma omp for collapse(4) schedule(static)
                for(int p = 0; p < CC.POLAR_DIM; p++){
                    for(int s = 0; s < CC.NSPEC; s++){
                        for(int j1 = 0; j1 < CC.NPOLE; j1++){
                            for(int j2 = 0; j2 < CC.NAZIM; j2++){
                                const double factor_dt = dt_inv - non_gray_smrt.getInvKn(p, s);
                                const double invKn = non_gray_smrt.getInvKn(p, s);
                                const double Cwp = non_gray_smrt.getHeatCap(p, s);
                                const double vg = non_gray_smrt.getVg(p, s);
                                const auto dir = dsa.dir()[j1][j2];
                                
                                const auto& approx_coeff = non_gray_smrt.getApproxCoeff(p, s, j1, j2);
                                
                                for(int k = 0; k < CC.N_MESH_CELL; k++){
                                    int CellID = comp_odr[j1][j2][k];
                                    // Build source term on-the-fly to avoid extra matrices:
                                    // rhs = invKn * Cwp / OMEGA * (MassMatT * Tc_col)
                                    rhs.noalias() = (invKn * Cwp / CC.OMEGA) * (MassMatT[CellID] * non_gray_smrt.getTc().col(CellID));
                                    rhs.noalias() += factor_dt * (MassMatT[CellID] * approx_coeff.col(CellID));
                                    // boundary condition
                                    for(int il = 0; il < dim+1; il++){
                                        int BCTAG = Cells[CellID]->getFaces()[il]->getBoundaryTag();
                                        double f = outflow[j1][j2](il, CellID);
                                        if(BCTAG == 0){
                                            int NBR = (Cells[CellID]->getFaces()[il]->getAdjacentCells()[0] == CellID) ? Cells[CellID]->getFaces()[il]->getAdjacentCells()[1] : Cells[CellID]->getFaces()[il]->getAdjacentCells()[0];
                                            rhs.noalias() -= 0.5 * vg * (f - abs(f)) * (FluxInt[CellID][il] * approx_coeff.col(NBR));
                                        }
                                        else if(CC.BOUNDARY_COND[BCTAG].first == 1){ // Thermalizing boundary condition
                                            rhs.noalias() -= 0.5 * vg * (f - abs(f)) * Cwp / CC.OMEGA * CC.BOUNDARY_COND[BCTAG].second * FluxInt[CellID][il].col(0);
                                        }
                                        else {
                                            throw std::runtime_error("Invalid boundary condition");
                                        }
                                    }
                                    const auto& lu = PreComputedLU[p][s][j1][j2][CellID];
                                    sol = lu.solve(rhs);
                                    non_gray_smrt.updateApproxCoeff(p, s, j1, j2, CellID, sol);
                                }
                            }
                        }
                    }
                }
            }

            res = non_gray_smrt.update(dsa, IntMat);
        
            std::cout << "Step " << step << "; Residual = " << res << std::endl;
            std::cout << "Tv = " << non_gray_smrt.getTv().norm() << std::endl;
            // Save step and res to file
            step_res_file << step << " " << res << std::endl;
            step++;
        }
    }

}

template class DGSolver::PBTE_NonGraySMRT<2>;
template class DGSolver::PBTE_NonGraySMRT<3>;