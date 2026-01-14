#include "Polynomial/Polynomial.hpp"
#include <stdexcept>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <numeric>

namespace Polynomial {

    // Helper function to compute binomial coefficient
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

    // Helper function to compute number of terms in a multivariate polynomial
    int computeNumTerms(int num_vars, int order) {
        return binomial(num_vars + order, order);
    }

    // Utility functions
    int Polynomial::getDoF() const {
        return computeNumTerms(num_vars_, order_);
    }

    std::vector<int> Polynomial::getExponents(int term_index) const{
        if (term_index < 0 || term_index >= getDoF()) {
            throw std::out_of_range("Term index out of range");
        }
        std::vector<int> exp(num_vars_, 0);
        if (term_index == 0){ return exp;}
        
        // Find the total degree of the term
        int sum_deg = 0;
        while (binomial(num_vars_ + sum_deg, sum_deg) <= term_index) {
            sum_deg ++;
        }

        int local_id = term_index - binomial(num_vars_ + sum_deg - 1, sum_deg - 1);
        for(int i = 0; i < num_vars_; i++){
            if(local_id == 0){
                exp[i] = sum_deg;
                break;
            }
            int rn = num_vars_ - i - 1;
            int rd = 0;
            while (binomial(rn + rd, rd) <= local_id){
                rd++;
            }
            exp[i] = sum_deg - rd;
            local_id -= binomial(rn + rd - 1, rd - 1);
            sum_deg = rd;
        }
        
        return exp;
    }

    int Polynomial::getIndex(const std::vector<int>& exponents) const{
        if (exponents.size() != num_vars_) {
            throw std::invalid_argument("Number of exponents does not match number of variables");
        }
        
        // Calculate total degree
        int sum_deg = std::accumulate(exponents.begin(), exponents.end(), 0);
        
        // If all exponents are 0, return 0
        if (sum_deg == 0) return 0;
        
        // Calculate the base index for this degree
        int index = binomial(num_vars_ + sum_deg - 1, sum_deg - 1);
        
        // Calculate the local index within this degree
        int local_id = 0;
        int remaining_deg = sum_deg;
        
        for (int i = 0; i < num_vars_; i++) {
            if (exponents[i] == sum_deg) {
                break;
            }
            int rn = num_vars_ - i - 1;
            int rd = remaining_deg - exponents[i];
            local_id += binomial(rn + rd - 1, rd - 1);
            remaining_deg = rd;
        }
        
        return index + local_id;
    }

    std::string Polynomial::toString(int precision) const {
        if (num_vars_ == 0) {
            return "0";
        }

        std::stringstream ss;
        ss << std::fixed << std::setprecision(precision);
        bool first_term = true;

        // Variable names
        std::vector<std::string> var_names = {"x", "y", "z"};

        for (int i = 0; i < coeffs_.size(); i++) {
            double coeff = coeffs_(i);
            if (std::abs(coeff) < 1e-10) continue;

            // Handle sign
            if (!first_term) {
                ss << (coeff >= 0 ? " + " : " - ");
            } else if (coeff < 0) {
                ss << "-";
            }
            first_term = false;

            // Print coefficient if it's not 1 or if it's not the constant term
            std::vector<int> powers = getExponents(i);
            bool is_constant = std::all_of(powers.begin(), powers.end(), [](int p) { return p == 0; });
            if (std::abs(std::abs(coeff) - 1.0) > 1e-10 || is_constant) {
                ss << std::abs(coeff);
            }

            // Print variables with their powers
            for (int j = 0; j < num_vars_; j++) {
                if (powers[j] > 0) {
                    ss << var_names[j];
                    if (powers[j] > 1) {
                        ss << "^" << powers[j];
                    }
                }
            }
        }

        if (first_term) {
            ss << "0";
        }

        return ss.str();
    }

    // Constructors
    Polynomial::Polynomial() : num_vars_(0), order_(0) {}

    Polynomial::Polynomial(int num_vars, int order) 
        : num_vars_(num_vars), order_(order) {
        if (num_vars < 0 || order < 0) {
            throw std::invalid_argument("Number of variables and order must be non-negative");
        }
        coeffs_ = Eigen::VectorXd::Zero(computeNumTerms(num_vars, order));
    }

    Polynomial::Polynomial(int num_vars, const Eigen::VectorXd& coeffs)
        : num_vars_(num_vars), coeffs_(coeffs) {
        if (num_vars < 0) {
            throw std::invalid_argument("Number of variables must be non-negative");
        }
        // Find the order by checking the size of coefficients
        order_ = 0;
        while (computeNumTerms(num_vars, order_) < coeffs.size()) {
            order_++;
        }
        if (computeNumTerms(num_vars, order_) != coeffs.size()) {
            throw std::invalid_argument("Invalid number of coefficients for given number of variables");
        }
    }

    // Setters
    void Polynomial::setCoefficients(const Eigen::VectorXd& coeffs) {
        if (coeffs.size() != computeNumTerms(num_vars_, order_)) {
            throw std::invalid_argument("Invalid number of coefficients");
        }
        coeffs_ = coeffs;
    }

    // Arithmetic operations
    Polynomial& Polynomial::operator=(const Polynomial& other) {
        if (this != &other) {  // Self-assignment check
            // Resize if necessary
            if (num_vars_ != other.num_vars_ || order_ != other.order_) {
                num_vars_ = other.num_vars_;
                order_ = other.order_;
                coeffs_.resize(other.coeffs_.size());
            }
            coeffs_ = other.coeffs_;
        }
        return *this;
    }

    Polynomial Polynomial::operator*(double scalar) const {
        Polynomial result = *this;
        result.coeffs_ *= scalar;
        return result;
    }

    Polynomial& Polynomial::operator*=(double scalar) {
        coeffs_ *= scalar;
        return *this;
    }

    Polynomial operator*(double scalar, const Polynomial& p) {
        return p * scalar;
    }

    Polynomial Polynomial::operator+(const Polynomial& other) const {
        if (num_vars_ != other.num_vars_) {
            throw std::runtime_error("Cannot add polynomials with different number of variables");
        }
        Polynomial result(num_vars_, std::max(order_, other.order_));
        result.coeffs_.head(coeffs_.size()) = coeffs_;
        result.coeffs_.head(other.coeffs_.size()) += other.coeffs_;
        return result;
    }

    Polynomial Polynomial::operator-(const Polynomial& other) const {
        if (num_vars_ != other.num_vars_) {
            throw std::runtime_error("Cannot subtract polynomials with different number of variables");
        }
        Polynomial result(num_vars_, std::max(order_, other.order_));
        result.coeffs_.head(coeffs_.size()) = coeffs_;
        result.coeffs_.head(other.coeffs_.size()) -= other.coeffs_;
        return result;
    }

    Polynomial Polynomial::operator*(const Polynomial& other) const {
        if (num_vars_ != other.num_vars_) {
            throw std::runtime_error("Cannot multiply polynomials with different number of variables");
        }
        Polynomial result(num_vars_, order_ + other.order_);
        
        // For each term in this polynomial
        for (int i = 0; i < coeffs_.size(); i++) {
            std::vector<int> powers_i = getExponents(i);
            
            // For each term in other polynomial
            for (int j = 0; j < other.coeffs_.size(); j++) {
                std::vector<int> powers_j = other.getExponents(j);
                
                // Compute product of monomials
                std::vector<int> product_powers(num_vars_);
                for (int k = 0; k < num_vars_; k++) {
                    product_powers[k] = powers_i[k] + powers_j[k];
                }
                
                // Add to result
                int result_idx = getIndex(product_powers);
                result.coeffs_(result_idx) += coeffs_(i) * other.coeffs_(j);
            }
        }
        
        return result;
    }

    Polynomial& Polynomial::operator+=(const Polynomial& other) {
        *this = *this + other;
        return *this;
    }

    Polynomial& Polynomial::operator-=(const Polynomial& other) {
        *this = *this - other;
        return *this;
    }

    Polynomial& Polynomial::operator*=(const Polynomial& other) {
        *this = *this * other;
        return *this;
    }

    // Evaluation
    double Polynomial::evaluate(const Eigen::VectorXd& point) const {
        if (point.size() != num_vars_) {
            throw std::runtime_error("Point dimension does not match number of variables");
        }
        
        double result = 0.0;
        for (int i = 0; i < coeffs_.size(); i++) {
            std::vector<int> powers = getExponents(i);
            double term = coeffs_(i);
            for (int j = 0; j < num_vars_; j++) {
                term *= std::pow(point[j], powers[j]);
            }
            result += term;
        }
        return result;
    }

    Eigen::VectorXd Polynomial::evaluateBatch(const Eigen::MatrixXd& points) const {
        if (points.rows() != num_vars_) {
            throw std::runtime_error("Points dimension does not match number of variables");
        }
        
        Eigen::VectorXd results(points.cols());
        for (int i = 0; i < points.cols(); i++) {
            results(i) = evaluate(points.col(i));  // Transpose the row vector to get a column vector
        }
        return results;
    }


    Polynomial Polynomial::derivative(int var_index) const {
        if (var_index < 0 || var_index >= num_vars_) {
            throw std::invalid_argument("Invalid variable index");
        }
        int result_order = std::max(0, order_ - 1);
        Polynomial result(num_vars_, result_order);
        for (int i = 0; i < coeffs_.size(); i++) {
            std::vector<int> powers = getExponents(i);
            if (powers[var_index] > 0) {
                std::vector<int> new_powers = powers;
                new_powers[var_index]--;
                int new_idx = result.getIndex(new_powers);
                result.coeffs_(new_idx) += coeffs_(i) * powers[var_index];
            }
        }
        return result;
    }

} // namespace Polynomial 