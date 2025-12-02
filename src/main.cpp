#include <iostream>
#include <vector>

#include "dg_solver.hpp"
#include <mfem.hpp>

int main() {
    std::cout << "DG4PBTE demo: basic build and clangd test\n";

    std::vector<double> values = {1.0, 2.0, 3.0, 4.0};
    double l2 = computeL2Norm(values);

    std::cout << "L2 norm of {1,2,3,4} = " << l2 << "\n";
    
    // Minimal MFEM sanity check: build a tiny mesh, finite element space, and project a constant
    try {
        mfem::Mesh mesh = mfem::Mesh::MakeCartesian1D(4);
        mfem::H1_FECollection fec(1, mesh.Dimension());
        mfem::FiniteElementSpace fes(&mesh, &fec);
        mfem::GridFunction u(&fes);
        mfem::ConstantCoefficient one(1.0);
        u.ProjectCoefficient(one);
        std::cout << "MFEM OK: dim=" << mesh.Dimension()
                  << ", elements=" << mesh.GetNE()
                  << ", dofs=" << fes.GetNDofs()
                  << ", ||u||_L2=" << u.Norml2() << "\n";
    } catch (const std::exception &e) {
        std::cerr << "MFEM test failed: " << e.what() << "\n";
        return 1;
    }
    return 0;
}


