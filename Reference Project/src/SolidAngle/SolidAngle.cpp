#include "SolidAngle/SolidAngle.hpp"
#include <iostream>
#include "Eigen/Dense"
#include "Utility/math_utils.hpp"

using namespace Eigen;
using namespace std;
using namespace Utility;

SolidAngle::SolidAngle(int dim, int npole, int nazim, int pattern)
{
    std::cout << ">>> Initialize the discrete solid angle ..." << std::endl;

    if((dim != 2 && dim != 3) || (pattern != 1 && pattern != 2)){
        throw std::invalid_argument("Wrong parameters for solid angle discretization");
    }

    if(pattern == 1){
        if(dim == 2){
            if(nazim % 4 != 0){
                throw std::invalid_argument("NPOLE % 2 != 0 or NAZIM % 4 != 0.");
            }
            int nazim_ = nazim / 4;
            direction.resize(npole);
            weight.resize(npole);
            for(int i = 0; i < npole; i++){
                direction[i].resize(nazim);
                weight[i].resize(nazim);
                for(int j = 0; j < nazim; j++){
                    direction[i][j].resize(dim);
                }
            }

            Eigen::MatrixXd GQ_phi = Utility::GaussQuad(0.0, M_PI / 2.0, nazim_);
            std::vector<double> cos_phi, sin_phi, w_phi;
            // azim
            for(int idx = 0; idx < nazim_; idx++){
                cos_phi.push_back(cos(GQ_phi(0, idx)));
                sin_phi.push_back(sin(GQ_phi(0, idx)));
                w_phi.push_back(GQ_phi(1, idx));
            }
            for(int idx = nazim_ - 1; idx >=0; idx--){
                cos_phi.push_back(-cos(GQ_phi(0, idx)));
                sin_phi.push_back(sin(GQ_phi(0, idx)));
                w_phi.push_back(GQ_phi(1, idx));
            }
            for(int idx = 0; idx < nazim_; idx++){
                cos_phi.push_back(-cos(GQ_phi(0, idx)));
                sin_phi.push_back(-sin(GQ_phi(0, idx)));
                w_phi.push_back(GQ_phi(1, idx));
            }
            for(int idx = nazim_ - 1; idx >=0; idx--){
                cos_phi.push_back(cos(GQ_phi(0, idx)));
                sin_phi.push_back(-sin(GQ_phi(0, idx)));
                w_phi.push_back(GQ_phi(1, idx));
            }

            // aggregation
            double weight_sum;
            for(int idx_t = 0; idx_t < npole; idx_t++){
                weight_sum = 0.0;
                for(int idx_p = 0; idx_p < nazim; idx_p++){
                    direction[idx_t][idx_p](0) = cos_phi[idx_p];
                    direction[idx_t][idx_p](1) = sin_phi[idx_p];
                    weight[idx_t][idx_p] = w_phi[idx_p];
                    weight_sum += weight[idx_t][idx_p];
                }
            }

            std::cout << "  >>> The Solid Angle Discritization is done! (sum_weight = " << weight_sum << ".)" << std::endl;
        }
        else if(dim == 3){
            if(npole % 2 != 0 || nazim % 4 != 0){
                throw std::invalid_argument("NPOLE % 2 != 0 or NAZIM % 4 != 0.");
            }
            int npole_ = npole / 2;
            int nazim_ = nazim / 4;
            direction.resize(npole);
            weight.resize(npole);
            for(int i = 0; i < npole; i++){
                direction[i].resize(nazim);
                weight[i].resize(nazim);
                for(int j = 0; j < nazim; j++){
                    direction[i][j].resize(dim);
                }
            }

            Eigen::MatrixXd GQ_theta = Utility::GaussQuad(-1.0, 0.0, npole_);
            Eigen::MatrixXd GQ_phi = Utility::GaussQuad(0.0, M_PI / 2.0, nazim_);
            std::vector<double> cos_theta, sin_theta, cos_phi, sin_phi, w_theta, w_phi;

            // pole
            for(int idx = 0; idx < npole_; idx++){
                cos_theta.push_back(-GQ_theta(0, idx));
                sin_theta.push_back(pow(1.0 - cos_theta[idx] * cos_theta[idx], 0.5));
                w_theta.push_back(GQ_theta(1, idx));
            }
            for(int idx = npole_ - 1; idx >= 0; idx--){
                cos_theta.push_back(GQ_theta(0, idx));
                sin_theta.push_back(pow(1.0 - cos_theta[idx] * cos_theta[idx], 0.5));
                w_theta.push_back(GQ_theta(1, idx));
            }
            // azim
            for(int idx = 0; idx < nazim_; idx++){
                cos_phi.push_back(cos(GQ_phi(0, idx)));
                sin_phi.push_back(sin(GQ_phi(0, idx)));
                w_phi.push_back(GQ_phi(1, idx));
            }
            for(int idx = nazim_ - 1; idx >=0; idx--){
                cos_phi.push_back(-cos(GQ_phi(0, idx)));
                sin_phi.push_back(sin(GQ_phi(0, idx)));
                w_phi.push_back(GQ_phi(1, idx));
            }
            for(int idx = 0; idx < nazim_; idx++){
                cos_phi.push_back(-cos(GQ_phi(0, idx)));
                sin_phi.push_back(-sin(GQ_phi(0, idx)));
                w_phi.push_back(GQ_phi(1, idx));
            }
            for(int idx = nazim_ - 1; idx >=0; idx--){
                cos_phi.push_back(cos(GQ_phi(0, idx)));
                sin_phi.push_back(-sin(GQ_phi(0, idx)));
                w_phi.push_back(GQ_phi(1, idx));
            }

            // aggregation
            double weight_sum = 0.0;
            for(int idx_t = 0; idx_t < npole; idx_t++){
                for(int idx_p = 0; idx_p < nazim; idx_p++){
                    direction[idx_t][idx_p](0) = sin_theta[idx_t] * cos_phi[idx_p];
                    direction[idx_t][idx_p](1) = sin_theta[idx_t] * sin_phi[idx_p];
                    direction[idx_t][idx_p](2) = cos_theta[idx_t];
                    weight[idx_t][idx_p] = w_theta[idx_t] * w_phi[idx_p];
                    weight_sum += weight[idx_t][idx_p];
                }
            }

            std::cout << "  >>> The Solid Angle Discritization is done! (sum_weight = " << weight_sum << ".)" << std::endl;
        }
    }
    else if (pattern == 2){
        if(dim == 2){
            direction.resize(npole);
            weight.resize(npole);
            for(int i = 0; i < npole; i++){
                direction[i].resize(nazim);
                weight[i].resize(nazim);
                for(int j = 0; j < nazim; j++){
                    direction[i][j].resize(dim);
                }
            }
            VectorXd Phi = VectorXd::Zero(nazim);
            VectorXd wPhi = VectorXd::Zero(nazim);
            MatrixXd GQ_phi = Utility::GaussQuad(0, M_PI, nazim / 2);
            Phi.segment(0, nazim / 2) = GQ_phi.row(0);
            wPhi.segment(0, nazim / 2) = GQ_phi.row(1);
            GQ_phi = Utility::GaussQuad(M_PI, 2.0 * M_PI, nazim / 2);
            Phi.segment(nazim / 2, nazim / 2) = GQ_phi.row(0).reverse();
            wPhi.segment(nazim / 2, nazim / 2) = GQ_phi.row(1).reverse();

            double weight_sum;
            for(int i = 0; i < npole; i++){
                weight_sum = 0.0;
                for(int j = 0; j < nazim; j++){
                    direction[i][j](0) = cos(Phi(j));
                    direction[i][j](1) = sin(Phi(j));
                    weight[i][j] = wPhi(j);
                    weight_sum += weight[i][j];
                }
            }

            std::cout << "  >>> The Solid Angle Discritization is done! (sum_weight = " << weight_sum << ".)" << std::endl;
        }
        else if(dim == 3){
            direction.resize(npole);
            weight.resize(npole);
            for(int i = 0; i < npole; i++){
                direction[i].resize(nazim);
                weight[i].resize(nazim);
                for(int j = 0; j < nazim; j++){
                    direction[i][j].resize(dim);
                }
            }

            MatrixXd GQ_the = GaussQuad(0, M_PI, npole);
            VectorXd Theta = GQ_the.row(0);
            VectorXd wTheta = GQ_the.row(1);

            VectorXd Phi = VectorXd::Zero(nazim);
            VectorXd wPhi = VectorXd::Zero(nazim);
            MatrixXd GQ_phi = Utility::GaussQuad(0, M_PI, nazim / 2);
            Phi.segment(0, nazim / 2) = GQ_phi.row(0);
            wPhi.segment(0, nazim / 2) = GQ_phi.row(1);
            GQ_phi = Utility::GaussQuad(M_PI, 2.0 * M_PI, nazim / 2);
            Phi.segment(nazim / 2, nazim / 2) = GQ_phi.row(0);
            wPhi.segment(nazim / 2, nazim / 2) = GQ_phi.row(1);

            double weight_sum = 0.0;
            for(int i = 0; i < npole; i++){
                for(int j = 0; j < nazim; j++){
                    direction[i][j](0) = sin(Theta(i)) * cos(Phi(j));
                    direction[i][j](1) = sin(Theta(i)) * sin(Phi(j));
                    direction[i][j](2) = cos(Theta(i));
                    weight[i][j] = sin(Theta(i)) * wTheta(i) * wPhi(j);
                    weight_sum += weight[i][j];
                }
            }

            std::cout << "  >>> The Solid Angle Discritization is done! (sum_weight = " << weight_sum << ".)" << std::endl;
        }
    }
}

SolidAngle::~SolidAngle(){}

std::vector<std::vector<Eigen::VectorXd>> SolidAngle::dir() const {return direction;}

std::vector<std::vector<double>> SolidAngle::wt() const { return weight;}

void SolidAngle::output() {
    std::cout << "***********************************SolidAngle***********************************" << std::endl;
    for(int i = 0; i < direction.size(); i++){
        for(int j = 0; j < direction[0].size(); j++){
            std::cout << "dsa_pts[" << i << "][" << j << "]" << std::endl;
            for(int k = 0; k < direction[0][0].size(); k++){
                std::cout << "crds[" << k << "]: " << direction[i][j](k) << " ";
            }
            std::cout << "wts: " << weight[i][j] << std::endl;
        }
    }
    std::cout << "***********************************SolidAngle***********************************" << std::endl;
}