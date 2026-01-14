#include "Utility/math_utils.hpp"
#include <vector>
#include "Eigen/Dense"

namespace Utility {
    
    /**
     * @brief Compute Gauss quadrature points and weights
     * @param a The lower bound of the interval
     * @param b The upper bound of the interval
     * @param N The number of quadrature points
     * @return The Gauss quadrature points and weights
     */
    Eigen::MatrixXd GaussQuad(double a, double b, int N){
        double alpha = 0.0, beta = 0.0;
        Eigen::MatrixXd GQ(2, N);
        if(N == 1){
            GQ(0, 0) = (alpha - beta) / (alpha + beta +2);
            GQ(1, 0) = 2;
            return GQ;
        }
        // Form symmetric matrix from recurrence.
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(N, N);
        Eigen::VectorXd h = Eigen::VectorXd::LinSpaced(N, 0, N-1);
        h = 2.0 * h.array() + alpha + beta;
        if(alpha + beta < 1e-15){
            J(0, 0) = 0.0;
        }
        else{
            J(0, 0) = -0.5 * (pow(alpha, 2) - pow(beta, 2)) / (h(0)+2.0) / h(0);
        }
        for(int i = 1; i < N; i++){
            J(i, i) = -0.5 * (pow(alpha, 2) - pow(beta, 2)) / (h(i)+2.0) / h(i);
            J(i-1, i) = 2.0 / (h(i-1)+2.0) * sqrt(i * (i + alpha + beta) * (i + alpha) * (i + beta) / (h(i-1) + 1.0) / (h(i-1) + 3.0));
        }
        Eigen::MatrixXd J_ = J.transpose();
        J = J.array() + J_.array();
        Eigen::EigenSolver<Eigen::MatrixXd> eig(J);
        GQ.row(0) = eig.eigenvalues().real();
        GQ.row(1) = eig.eigenvectors().real().row(0).array().pow(2) * pow(2.0, alpha + beta + 1) / (alpha + beta + 1) 
                    * tgamma(alpha + 1) * tgamma(beta + 1) / tgamma(alpha + beta + 1);

        Eigen::MatrixXd GQ_sorted(2, N);
        Eigen::VectorXd vec = GQ.row(0);
        Eigen::VectorXi ind = Eigen::VectorXi::LinSpaced(vec.size(), 0, vec.size()-1);
        auto rule = [vec](int i, int j) -> bool{return vec(i) < vec(j);};
        std::sort(ind.data(), ind.data()+ind.size(), rule);
        for(int i = 0; i < vec.size(); i++){
            GQ_sorted.col(i) = GQ.col(ind(i));
        }

        Eigen::VectorXd x = GQ_sorted.row(0);
        Eigen::VectorXd w = GQ_sorted.row(1);
        GQ_sorted.row(0) = 0.5 * ((b-a) * x.array() + (a+b)) ;
        GQ_sorted.row(1) = 0.5 * (b-a) * w.array();

        return GQ_sorted;
    }

    /**
     * @brief Compute the factorial of a number
     * @param n The number
     * @return The factorial of the number
     */
    double Factorial(int n){
        if(n == 0){return 1.0;}
        return double(n) * Factorial(n-1);
    }

    /**
     * @brief Compute the integral of a monomial x^a * y^b * z^c or x^a * y^b
     * @param dim The dimension of the monomial
     * @param pow The power of the monomial {a, b, c} or {a, b}
     * @return The integral of the monomial
     */
    double int_splx_mono(int dim, std::vector<int> pow) {
        double result = 1.0;
        int N  = 0;
        switch (dim) {
            case 1:
                for(int i = 0; i < pow.size(); i++){
                    result *= Utility::Factorial(pow[i]);
                    N += pow[i];
                }
                result /= Utility::Factorial(N + 1);
                break;
            case 2:
                for(int i = 0; i < pow.size(); i++){
                    result *= Utility::Factorial(pow[i]);
                    N += pow[i];
                }
                result /= Utility::Factorial(N + 2);
                break;
            case 3:
                for(int i = 0; i < pow.size(); i++){
                    result *= Utility::Factorial(pow[i]);
                    N += pow[i];
                }
                result /= Utility::Factorial(N + 3);
                break;
            default:
                throw std::invalid_argument("int_splx_mono: Dimension must be 1, 2 or 3");
        }
        return result;
    }

    /**
     * @brief Compute the integral of the complete polynomial consisting of all monomials up to a given order on the reference simplex
     * @param dim The dimension of the reference simplex
     * @param order The order of the polynomial
     * @return The integral of the complete polynomial
     */
    Eigen::VectorXd int_splx_complete(int dim, int order) {
        Eigen::VectorXd IntEle;
        int idx = 0;
        switch (dim) {
            case 1:
                IntEle = Eigen::VectorXd::Zero(order + 1);
                for(int i = 0; i <= order; i++){
                    IntEle(idx) = int_splx_mono(1, {i});
                    idx++;
                }
                break;
            case 2:
                IntEle = Eigen::VectorXd::Zero((order + 1) * (order + 2) / 2);
                for(int d = 0; d <= order; d++){
                    for(int dy = 0; dy <= d; dy++){
                        int dx = d - dy;
                        IntEle(idx) = int_splx_mono(2, {dx, dy});
                        idx++;
                    }
                }
                break;
            case 3:
                IntEle = Eigen::VectorXd::Zero((order + 1) * (order + 2) * (order + 3) / 6);
                // for(int d = 0; d <= order; d++){
                //     for(int dz = 0; dz <= d; dz++){
                //         for(int dy = 0; dy <= d - dz; dy++){
                //             int dx = d - dz - dy;
                //             IntEle(idx) = int_splx_mono(3, {dx, dy, dz});
                //             idx++;
                //         }
                //     }
                // }
                for(int d = 0; d <= order; d++){
                    for(int dx = d; dx >= 0; dx--){
                        for(int dy = d-dx; dy >= 0; dy--){
                            int dz = d - dx - dy;
                            IntEle(idx) = int_splx_mono(3, {dx, dy, dz});
                            idx++;
                        }
                    }
                }
                break;
            default:
                throw std::invalid_argument("int_splx_complete: Dimension must be 1, 2 or 3");
        }
        return IntEle;
    }

    int binomial(int n, int k) {
        if (k < 0 || k > n) return 0;
        if (k == 0 || k == n) return 1;
        k = std::min(k, n - k);
        int c = 1;
        for (int i = 0; i < k; i++) {
            c = c * (n - i) / (i + 1);
        }
        return c;
    }

}