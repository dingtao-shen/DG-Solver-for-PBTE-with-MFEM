#include <iostream>
#include <vector>

#include "dg_solver.hpp"

int main() {
    std::cout << "DG4PBTE demo: basic build and clangd test\n";

    std::vector<double> values = {1.0, 2.0, 3.0, 4.0};
    double l2 = computeL2Norm(values);

    std::cout << "L2 norm of {1,2,3,4} = " << l2 << "\n";
    return 0;
}


