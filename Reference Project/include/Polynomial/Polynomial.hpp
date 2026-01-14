#pragma once

#include <vector>
#include <string>
#include "Eigen/Dense"

namespace Polynomial {

    class Polynomial {
    private:

        int num_vars_;  // Number of variables
        int order_;     // Order of the polynomial
        Eigen::VectorXd coeffs_;  // Coefficients stored in a vector

    public:

        // Utility functions
        int getDoF() const;  // Get number of terms in the polynomial
        std::vector<int> getExponents(int term_index) const;  // Get powers of variables for a term
        int getIndex(const std::vector<int>& exponents) const;  // Get index of a term given its powers
        std::string toString(int precision = 6) const;  // Convert polynomial to string representation

        // Constructors
        Polynomial();  // Default constructor
        Polynomial(int num_vars, int order);  // Constructor with number of variables and order
        Polynomial(int num_vars, const Eigen::VectorXd& coeffs);  // Constructor with coefficients and number of variables

        // Setters
        void setCoefficients(const Eigen::VectorXd& coeffs);  // Set coefficients
        // Getters
        int getNumVars() const { return num_vars_; }
        int getOrder() const { return order_; }
        const Eigen::VectorXd& getCoefficients() const { return coeffs_; }

        // Arithmetic operations
        Polynomial& operator=(const Polynomial& other);  // Assignment operator
        Polynomial operator*(double scalar) const;  // Scalar multiplication
        Polynomial& operator*=(double scalar);  // Scalar multiplication assignment
        friend Polynomial operator*(double scalar, const Polynomial& p);  // Left-side scalar multiplication
        Polynomial operator+(const Polynomial& other) const;
        Polynomial operator-(const Polynomial& other) const;
        Polynomial operator*(const Polynomial& other) const;
        Polynomial& operator+=(const Polynomial& other);
        Polynomial& operator-=(const Polynomial& other);
        Polynomial& operator*=(const Polynomial& other);

        // Evaluation
        double evaluate(const Eigen::VectorXd& point) const;
        Eigen::VectorXd evaluateBatch(const Eigen::MatrixXd& points) const;

        // Derivatives
        Polynomial derivative(int var_index) const;  // Partial derivative with respect to variable var_index
        
    };

} // namespace Polynomial 