#pragma once

/* non-gray model with double mode time relaxation */

#include "PhononModel/PhonoModel.hpp"
#include "SolidAngle/SolidAngle.hpp"
#include "Eigen/Dense"
#include "SpatialMesh/SpatialMesh.hpp"
#include "SpatialMesh/MeshPartitioning.hpp"
#include <mpi.h>

namespace PhononModel {

    template<int SpatialDim>
    class NonGraySMRT : public PhononModel<SpatialDim> {
        protected:
            // parameters of the phonon model
            std::vector<Eigen::MatrixXd> W; // frequency of the phonon
            std::vector<Eigen::VectorXd> K; // wave vector of the phonon
            std::vector<Eigen::VectorXd> invKn; // inverse Knudsen number for normal scattering
            std::vector<Eigen::VectorXd> Vg; // group velocity
            std::vector<Eigen::VectorXd> D; // phonon state density
            std::vector<Eigen::VectorXd> HeatCap; // heat capacity
            double HeatCapV;
            // coefficients of the polynomial approximation
            std::vector<std::vector<std::vector<std::vector<Eigen::MatrixXd>>>> ApproxCoeff; // npolar * nspect * npole * nazim * (dof * ncell)
            std::vector<double> ApproxCoeff_Vec; // serialized version of ApproxCoeff npolar * nspect * npole * nazim * ncell * dof
            // physical fields
            Eigen::MatrixXd Tc; // weighted coefficients of basis functions in each cell to compute temperature
            Eigen::VectorXd Tv; // avaerage temperature in each cell
            std::vector<Eigen::MatrixXd> Qc; // weighted coefficients of basis functions in each cell to compute heat flux
            std::vector<Eigen::VectorXd> Qv; // avaerage heat flux in each cell
            std::vector<Eigen::MatrixXd> Uc; // drift velocity
        public:
            // constructors
            explicit NonGraySMRT();

            // update
            void updateApproxCoeff(int idx_polar, int idx_spec, int idx_pole, int idx_azim, int idx_cell, const Eigen::VectorXd& Solution);
            double update(const SolidAngle& dsa, const Eigen::MatrixXd& IntMat);
            double cal_residual_MPI(const Eigen::MatrixXd& IntMat);

            const std::vector<Eigen::MatrixXd>& getW() const { return W; }
            const Eigen::MatrixXd& getW(int idx_polar) const { return W[idx_polar]; }
            const double getW(int idx_polar, int idx_spec) const { return W[idx_polar](0, idx_spec); }

            const std::vector<Eigen::VectorXd>& getK() const { return K; }
            const Eigen::VectorXd& getK(int idx_polar) const { return K[idx_polar]; }
            const double getK(int idx_polar, int idx_spec) const { return K[idx_polar](idx_spec); }

            const std::vector<Eigen::VectorXd>& getInvKn() const { return invKn; }
            const Eigen::VectorXd& getInvKn(int idx_polar) const { return invKn[idx_polar]; }
            const double getInvKn(int idx_polar, int idx_spec) const { return invKn[idx_polar](idx_spec); }

            const std::vector<Eigen::VectorXd>& getVg() const { return Vg; }
            const Eigen::VectorXd& getVg(int idx_polar) const { return Vg[idx_polar]; }
            const double getVg(int idx_polar, int idx_spec) const { return Vg[idx_polar](idx_spec); }

            const std::vector<Eigen::VectorXd>& getD() const { return D; }
            const Eigen::VectorXd& getD(int idx_polar) const { return D[idx_polar]; }
            const double getD(int idx_polar, int idx_spec) const { return D[idx_polar](idx_spec); }

            const std::vector<Eigen::VectorXd>& getHeatCap() const { return HeatCap; }
            const Eigen::VectorXd& getHeatCap(int idx_polar) const { return HeatCap[idx_polar]; }
            const double getHeatCap(int idx_polar, int idx_spec) const { return HeatCap[idx_polar](idx_spec); }

            const double getHeatCapV() const { return HeatCapV; }

            const Eigen::MatrixXd& getTc() const { return Tc; }
            const Eigen::VectorXd& getTv() const { return Tv; }
            const std::vector<Eigen::MatrixXd>& getQc() const { return Qc; }
            const std::vector<Eigen::VectorXd>& getQv() const { return Qv; }
            const std::vector<Eigen::MatrixXd>& getUc() const { return Uc; }

            const std::vector<std::vector<std::vector<std::vector<Eigen::MatrixXd>>>>& getApproxCoeff() const { return ApproxCoeff; }
            const std::vector<std::vector<Eigen::MatrixXd>>& getApproxCoeff(int idx_polar, int idx_spec) const { return ApproxCoeff[idx_polar][idx_spec]; }
            const Eigen::MatrixXd& getApproxCoeff(int idx_polar, int idx_spec, int idx_pole, int idx_azim) const { return ApproxCoeff[idx_polar][idx_spec][idx_pole][idx_azim]; }

            // setters
            void setTcColumn(int idx_cell, const Eigen::VectorXd& col) { Tc.col(idx_cell) = col; }

            // output 
            void output_2D_slice_T_Q(std::vector<std::shared_ptr<SpatialMesh::Cell<2, 3>>>& Cells,
                SolidAngle& dsa, std::string results_dir);
            void output_3D_2Dslice_T_Q(std::vector<std::shared_ptr<SpatialMesh::Cell<3, 4>>>& Cells,
                                SolidAngle& dsa, std::string results_dir, int fix_crd_idx, double crd);
            void output_3D_1Dslice_T_Q(std::vector<std::shared_ptr<SpatialMesh::Cell<3, 4>>>& Cells,
                SolidAngle& dsa, std::string results_dir, int dir, double crd1, double crd2);
    };

    template<int SpatialDim>
    class NonGraySMRT_MPI : public PhononModel<SpatialDim> {
        protected:
            // parameters of the phonon model
            std::vector<Eigen::MatrixXd> W; // frequency of the phonon
            std::vector<Eigen::VectorXd> K; // wave vector of the phonon
            std::vector<Eigen::VectorXd> invKn; // inverse Knudsen number for normal scattering
            std::vector<Eigen::VectorXd> Vg; // group velocity
            std::vector<Eigen::VectorXd> D; // phonon state density
            std::vector<Eigen::VectorXd> HeatCap; // heat capacity
            double HeatCapV;

            int world_rank;
            int world_size;
            MPI_Comm comm;
            int n_local_cells;
            
            // coefficients of the polynomial approximation
            std::vector<double> local_ApproxCoeff_Vec; // serialized version of ApproxCoeff npolar * nspect * npole * nazim * ncell * dof
            // physical fields
            // weighted coefficients of basis functions in each cell to compute temperature
            std::vector<double> local_Tc_Vec;
            // avaerage temperature in each cell
            std::vector<double> local_Tv_Vec;
            // weighted coefficients of basis functions in each cell to compute heat flux
            std::vector<double> local_Qc_Vec;
            // avaerage heat flux in each cell
            std::vector<double> local_Qv_Vec;
            // drift velocity
            std::vector<double> local_Uc_Vec;

            std::vector<std::vector<std::vector<std::vector<std::vector<Eigen::VectorXd>>>>> global_ApproxCoeff; 
            Eigen::MatrixXd global_Tc;
            Eigen::VectorXd global_Tv;
            std::vector<Eigen::MatrixXd> global_Qc;
            std::vector<Eigen::VectorXd> global_Qv;
            std::vector<Eigen::MatrixXd> global_Uc;

        public:
            // constructors
            explicit NonGraySMRT_MPI(const SpatialMesh::MeshPartitionInfo<SpatialDim>& partition_info, MPI_Comm communicator = MPI_COMM_WORLD);

            // update
            void updateApproxCoeff(int idx_cell, int idx_polar, int idx_spec, int idx_pole, int idx_azim, const Eigen::VectorXd& Solution);
            void update_local_ApproxCoeff(int idx_cell, int idx_polar, int idx_spec, int idx_pole, int idx_azim, const Eigen::VectorXd& Solution, const SpatialMesh::MeshPartitionInfo<SpatialDim>& partition_info);
            void update_global_Tc(int cell_idx, const Eigen::VectorXd& cell_Tc);
            double cal_residual_MPI(const Eigen::MatrixXd& IntMat);

            // getters
            const std::vector<Eigen::MatrixXd>& getW() const { return W; }
            const Eigen::MatrixXd& getW(int idx_polar) const { return W[idx_polar]; }
            const double getW(int idx_polar, int idx_spec) const { return W[idx_polar](0, idx_spec); }

            const std::vector<Eigen::VectorXd>& getK() const { return K; }
            const Eigen::VectorXd& getK(int idx_polar) const { return K[idx_polar]; }
            const double getK(int idx_polar, int idx_spec) const { return K[idx_polar](idx_spec); }

            const std::vector<Eigen::VectorXd>& getInvKn() const { return invKn; }
            const Eigen::VectorXd& getInvKn(int idx_polar) const { return invKn[idx_polar]; }
            const double getInvKn(int idx_polar, int idx_spec) const { return invKn[idx_polar](idx_spec); }

            const std::vector<Eigen::VectorXd>& getVg() const { return Vg; }
            const Eigen::VectorXd& getVg(int idx_polar) const { return Vg[idx_polar]; }
            const double getVg(int idx_polar, int idx_spec) const { return Vg[idx_polar](idx_spec); }

            const std::vector<Eigen::VectorXd>& getD() const { return D; }
            const Eigen::VectorXd& getD(int idx_polar) const { return D[idx_polar]; }
            const double getD(int idx_polar, int idx_spec) const { return D[idx_polar](idx_spec); }

            const std::vector<Eigen::VectorXd>& getHeatCap() const { return HeatCap; }
            const Eigen::VectorXd& getHeatCap(int idx_polar) const { return HeatCap[idx_polar]; }
            const double getHeatCap(int idx_polar, int idx_spec) const { return HeatCap[idx_polar](idx_spec); }

            const double getHeatCapV() const { return HeatCapV; }

            const std::vector<double>& getLocalTc() const { return local_Tc_Vec; }
            const std::vector<double>& getLocalTv() const { return local_Tv_Vec; }
            const std::vector<double>& getLocalQc() const { return local_Qc_Vec; }
            const std::vector<double>& getLocalQv() const { return local_Qv_Vec; }
            const std::vector<double>& getLocalUc() const { return local_Uc_Vec; }

            const std::vector<std::vector<std::vector<std::vector<std::vector<Eigen::VectorXd>>>>>& getGlobalApproxCoeff() const { return global_ApproxCoeff; }
            const Eigen::MatrixXd& getGlobalTc() const { return global_Tc; }
            const Eigen::VectorXd& getGlobalTv() const { return global_Tv; }
            const std::vector<Eigen::MatrixXd>& getGlobalQc() const { return global_Qc; }
            const std::vector<Eigen::VectorXd>& getGlobalQv() const { return global_Qv; }
            const std::vector<Eigen::MatrixXd>& getGlobalUc() const { return global_Uc; }

            const std::vector<double>& getLocalApproxCoeffVec() const { return local_ApproxCoeff_Vec; }

            // output 
            void output_2D_slice_T_Q(std::vector<std::shared_ptr<SpatialMesh::Cell<2, 3>>>& Cells,
                SolidAngle& dsa, std::string results_dir);
            void output_3D_2Dslice_T_Q(std::vector<std::shared_ptr<SpatialMesh::Cell<3, 4>>>& Cells,
                                SolidAngle& dsa, std::string results_dir, int fix_crd_idx, double crd);
            void output_3D_1Dslice_T_Q(std::vector<std::shared_ptr<SpatialMesh::Cell<3, 4>>>& Cells,
                SolidAngle& dsa, std::string results_dir, int dir, double crd1, double crd2);
    };
}