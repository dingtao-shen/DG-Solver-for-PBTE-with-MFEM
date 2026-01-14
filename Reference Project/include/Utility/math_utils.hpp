#pragma once

#include <vector>
#include "Eigen/Dense"

namespace Utility {

    /**
     * @brief Gauss-Legendre Quadrature
     * @param a lower bound
     * @param b upper bound
     * @param N number of quadrature points
     * @return matrix of quadrature points and weights
     */
    Eigen::MatrixXd GaussQuad(double a, double b, int N);

    /**
     * @brief Compute the factorial of a number
     * @param n The number
     * @return The factorial of the number
     */
    double Factorial(int n);

    /**
     * @brief Compute the integral of a monomial x^a * y^b * z^c or x^a * y^b
     * @param dim The dimension of the monomial
     * @param pow The power of the monomial {a, b, c} or {a, b}
     * @return The integral of the monomial
     */
    double int_splx_mono(int dim, std::vector<int> pow);

    /**
     * @brief Compute the integral of the complete polynomial consisting of all monomials up to a given order on the reference simplex
     * @param dim The dimension of the reference simplex
     * @param order The order of the polynomial
     * @return The integral of the complete polynomial
     */
    Eigen::VectorXd int_splx_complete(int dim, int order);

    /**
     * @brief Compute the binomial coefficient
     * @param n The number
     * @param k The number
     * @return The binomial coefficient
     */
    int binomial(int n, int k);

}