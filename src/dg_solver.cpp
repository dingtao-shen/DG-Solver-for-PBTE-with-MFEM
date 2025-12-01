#include "dg_solver.hpp"

#include <cmath>

double computeL2Norm(const std::vector<double>& values) {
    double sumSquares = 0.0;
    for (double value : values) {
        sumSquares += value * value;
    }
    return std::sqrt(sumSquares);
}


