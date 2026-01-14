#pragma once

#include <vector>
#include <fstream>
#include <filesystem>
#include "Eigen/Dense"
#include "SpatialMesh/SpatialMesh.hpp"
#include "Polynomial/Polynomial.hpp"
#include "PolyFem/BasisFunctions.hpp"
#include "PolyFem/Quadrature.hpp"
#include "Utility/math_utils.hpp"
#include "GlobalConfig/GlobalConfig.hpp"

#include <iostream>
#include <omp.h>

namespace PolyFem{
    
    template<int dim>
    class Integral {
        static_assert(dim == 2 || dim == 3, "Mesh dimension must be 2 or 3");
        protected:
            Eigen::MatrixXd IntMat;
            std::vector<Eigen::MatrixXd> MassMat;
            std::vector<std::vector<Eigen::VectorXd>> IntFaceMat;
            std::vector<std::vector<Eigen::MatrixXd>> StiffMat;
            std::vector<std::vector<Eigen::MatrixXd>> MassFaceMat;
            std::vector<std::vector<Eigen::MatrixXd>> FluxMat;
        public:
            explicit Integral(const SpatialMesh::SpatialMesh<dim>& mesh);
            virtual ~Integral(){};

            virtual void updateDirichletFlux(const SpatialMesh::SpatialMesh<dim>& mesh, double dt);
            
            Eigen::MatrixXd getIntMat() const {return IntMat;}
            std::vector<std::vector<Eigen::VectorXd>> getIntFaceMat() const {return IntFaceMat;}
            std::vector<Eigen::MatrixXd> getMassMat() const {return MassMat;}
            std::vector<std::vector<Eigen::MatrixXd>> getStiffMat() const {return StiffMat;}
            std::vector<std::vector<Eigen::MatrixXd>> getMassFaceMat() const {return MassFaceMat;}
            std::vector<std::vector<Eigen::MatrixXd>> getFluxMat() const {return FluxMat;}
            void save(std::string results_dir);
    };

    template<int dim>
    Integral<dim>::Integral(const SpatialMesh::SpatialMesh<dim>& mesh){
        int Nt = mesh.getNumCells();
        std::vector<double> Volumes(Nt, 0.0);
        for(int i = 0; i < Nt; i++){
            Volumes[i] = mesh.getCells()[i]->getMeasure();
        }
        double alphaVolume, alphaFace;
        if(dim == 2){
            alphaVolume = 2.0;
            alphaFace = 1.0;
        }
        else if(dim == 3){
            alphaVolume = 6.0;
            alphaFace = 2.0;
        }
        
        std::vector<Polynomial::Polynomial> RefBasis = ReferenceBasis(dim, CC.POLYDEG);

        std::cout << ">>> Computing integrals for FEM elementwise operations..." << std::endl;

        /* Compute the integration matrix */
        IntMat.resize(Nt, CC.DOF);
        IntMat.setZero();
        Eigen::VectorXd IntEle = Utility::int_splx_complete(dim, CC.POLYDEG);
        double s = 0.0;
        for (int i = 0; i < CC.DOF; i++){
            s = RefBasis[i].getCoefficients().dot(IntEle);
            for (int k = 0; k < Nt; k++){ 
                IntMat(k, i) = s * alphaVolume * Volumes[k];
            }
        }
        std::cout << "  >>> IntMat computed." << std::endl;

        /* Compute the mass matrix */
        MassMat.resize(Nt);
        for(int i = 0; i < Nt; i++){
            MassMat[i].resize(CC.DOF, CC.DOF);
            MassMat[i].setZero();
        }
        Eigen::VectorXd IntEleMass = Utility::int_splx_complete(dim, CC.POLYDEG + CC.POLYDEG);
        for(int j = 0; j < CC.DOF; j++){
            for(int k = 0; k < CC.DOF; k++){
                Polynomial::Polynomial product = RefBasis[j] * RefBasis[k];
                for(int i = 0; i < Nt; i++){
                    MassMat[i](j, k) = product.getCoefficients().dot(IntEleMass) * alphaVolume * Volumes[i];
                }
            }
        }
        std::cout << "  >>> MassMat computed." << std::endl;

        /* Compute the stiffness matrix */
        StiffMat.resize(Nt);
        for(int i = 0; i < Nt; i++){
            StiffMat[i].resize(dim);
            for(int j = 0; j < dim; j++){
                StiffMat[i][j].resize(CC.DOF, CC.DOF);
                StiffMat[i][j].setZero();
            }
        }
        std::vector<Eigen::MatrixXd> refStifMat(dim, Eigen::MatrixXd::Zero(CC.DOF, CC.DOF));
        Eigen::VectorXd IntEleDer = Utility::int_splx_complete(dim, CC.POLYDEG + CC.POLYDEG - 1);
        s = 0.0;
        for(int j = 0; j < CC.DOF; j++){
            for(int k = 0; k < CC.DOF; k++){
                for(int l = 0; l < dim; l++){
                    refStifMat[l](j, k) = (RefBasis[j].derivative(l) * RefBasis[k]).getCoefficients().dot(IntEleDer);              
                }
                for(int i = 0; i < Nt; i++){
                    Eigen::MatrixXd JacobianMatrix = mesh.getCells()[i]->getJacobianMatrix();
                    double detJM = JacobianMatrix.determinant();
                    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(dim, dim);
                    if(dim == 2){
                        J(0, 0) = JacobianMatrix(1,1) / detJM;
                        J(0, 1) = -JacobianMatrix(1,0) / detJM;
                        J(1, 0) = -JacobianMatrix(0,1) / detJM;
                        J(1, 1) = JacobianMatrix(0,0) / detJM;
                    }
                    else if(dim == 3){
                        J(0, 0) = (JacobianMatrix(1,1) * JacobianMatrix(2,2) - JacobianMatrix(2,1) * JacobianMatrix(1,2)) / detJM;
                        J(0, 1) = - (JacobianMatrix(1,0) * JacobianMatrix(2,2) - JacobianMatrix(2,0) * JacobianMatrix(1,2)) / detJM;
                        J(0, 2) = (JacobianMatrix(1,0) * JacobianMatrix(2,1) - JacobianMatrix(2,0) * JacobianMatrix(1,1)) / detJM;
                        J(1, 0) = - (JacobianMatrix(0,1) * JacobianMatrix(2,2) - JacobianMatrix(2,1) * JacobianMatrix(0,2)) / detJM;
                        J(1, 1) = (JacobianMatrix(0,0) * JacobianMatrix(2,2) - JacobianMatrix(2,0) * JacobianMatrix(0,2)) / detJM;
                        J(1, 2) = - (JacobianMatrix(0,0) * JacobianMatrix(2,1) - JacobianMatrix(2,0) * JacobianMatrix(0,1)) / detJM;
                        J(2, 0) = (JacobianMatrix(0,1) * JacobianMatrix(1,2) - JacobianMatrix(1,1) * JacobianMatrix(0,2)) / detJM;
                        J(2, 1) = - (JacobianMatrix(0,0) * JacobianMatrix(1,2) - JacobianMatrix(1,0) * JacobianMatrix(0,2)) / detJM;
                        J(2, 2) = (JacobianMatrix(0,0) * JacobianMatrix(1,1) - JacobianMatrix(1,0) * JacobianMatrix(0,1)) / detJM;
                    }
                    for(int l = 0; l < dim; l++){
                        for(int m = 0; m < dim; m++){
                            StiffMat[i][l](j, k) += J(l, m) * refStifMat[m](j, k) * alphaVolume * Volumes[i];
                        }
                    }
                }
            }
        }
        std::cout << "  >>> StiffMat computed." << std::endl;


        /* Compute the integration matrix on the faces */
        IntFaceMat.resize(Nt);
        for(int i = 0; i < Nt; i++){
            IntFaceMat[i].resize(dim+1);
            for(int j = 0; j < dim+1; j++){
                IntFaceMat[i][j].resize(CC.DOF);
                IntFaceMat[i][j].setZero();
            }
        }
        std::vector<Eigen::VectorXd> IntEleFace = std::vector<Eigen::VectorXd>(dim+1, Eigen::VectorXd::Zero(CC.DOF));
        int idx = 0;
        if(dim == 2){
            for(int d = 0; d <= CC.POLYDEG; d++){
                for(int dy = 0; dy <= d; dy++){
                    int dx = d - dy;
                    if(dx == 0){
                        IntEleFace[2](idx) = Utility::int_splx_mono(1, {dy});
                    }
                    if(dy == 0){
                        IntEleFace[0](idx) = Utility::int_splx_mono(1, {dx});
                    }
                    IntEleFace[1](idx) = Utility::int_splx_mono(1, {dx, dy});
                    idx ++;
                }
            }
        }
        else if(dim == 3){
            for(int d = 0; d <= CC.POLYDEG; d++){
                for(int dx = d; dx >= 0; dx--){
                    for(int dy = d-dx; dy >= 0; dy--){
                        int dz = d - dx - dy;
                        if(dx == 0){
                            IntEleFace[2](idx) = Utility::int_splx_mono(2, {dy, dz});
                        }
                        if(dy == 0){
                            IntEleFace[3](idx) = Utility::int_splx_mono(2, {dx, dz});
                        }
                        if(dz == 0){
                            IntEleFace[0](idx) = Utility::int_splx_mono(2, {dx, dy});
                        }
                        IntEleFace[1](idx) = Utility::int_splx_mono(2, {dx, dy, dz});
                        idx ++;
                    }
                }
            }
        }
        for(int j = 0; j < CC.DOF; j++){
            for(int i = 0; i < Nt; i++){
                for(int l = 0; l < dim + 1; l++){
                    IntFaceMat[i][l](j) = RefBasis[j].getCoefficients().dot(IntEleFace[l]) * alphaFace * mesh.getCells()[i]->getFaces()[l]->getMeasure();
                }
            }
        }
        std::cout << "  >>> IntFaceMat computed." << std::endl;

        /* Compute the mass matrix on the faces */
        MassFaceMat.resize(Nt);
        for(int i = 0; i < Nt; i++){
            MassFaceMat[i].resize(dim+1);
            for(int j = 0; j < dim+1; j++){
                MassFaceMat[i][j].resize(CC.DOF, CC.DOF);
                MassFaceMat[i][j].setZero();
            }
        }
        int dof2;
        if(dim == 2){
            dof2 = (CC.POLYDEG + CC.POLYDEG + 1) * (CC.POLYDEG + CC.POLYDEG + 2) / 2;
        }
        else if(dim == 3){
            dof2 = (CC.POLYDEG + CC.POLYDEG + 1) * (CC.POLYDEG + CC.POLYDEG + 2) * (CC.POLYDEG + CC.POLYDEG + 3) / 6;
        }
        std::vector<Eigen::VectorXd> IntEleMassFace = std::vector<Eigen::VectorXd>(dim+1, Eigen::VectorXd::Zero(dof2));
        idx = 0;
        if(dim == 2){
            for(int d = 0; d <= CC.POLYDEG + CC.POLYDEG; d++){
                for(int dy = 0; dy <= d; dy++){
                    int dx = d - dy;
                    if(dx == 0){
                        IntEleMassFace[2](idx) = Utility::int_splx_mono(1, {dy});
                    }
                    if(dy == 0){
                        IntEleMassFace[0](idx) = Utility::int_splx_mono(1, {dx});
                    }
                    IntEleMassFace[1](idx) = Utility::int_splx_mono(1, {dx, dy});
                    idx ++;
                }
            }
        }
        else if(dim == 3){
            for(int d = 0; d <= CC.POLYDEG + CC.POLYDEG; d++){
                for(int dx = d; dx >= 0; dx--){
                    for(int dy = d-dx; dy >= 0; dy--){
                        int dz = d - dx - dy;
                        if(dx == 0){
                            IntEleMassFace[2](idx) = Utility::int_splx_mono(2, {dy, dz});
                        }
                        if(dy == 0){
                            IntEleMassFace[3](idx) = Utility::int_splx_mono(2, {dx, dz});
                        }
                        if(dz == 0){
                            IntEleMassFace[0](idx) = Utility::int_splx_mono(2, {dx, dy});
                        }
                        IntEleMassFace[1](idx) = Utility::int_splx_mono(2, {dx, dy, dz});
                        idx ++;
                    }
                }
            }
        }
        for(int j = 0; j < CC.DOF; j++){
            for(int k = 0; k < CC.DOF; k++){
                Polynomial::Polynomial product = RefBasis[j] * RefBasis[k];
                for(int i = 0; i < Nt; i++){
                    for(int l = 0; l < dim + 1; l++){
                        MassFaceMat[i][l](j, k) = product.getCoefficients().dot(IntEleMassFace[l]) * alphaFace * mesh.getCells()[i]->getFaces()[l]->getMeasure();
                    }
                }
            }
        }
        std::cout << "  >>> MassFaceMat computed." << std::endl;

        /* Compute the product of basis functions from both sides of the faces */
        FluxMat.resize(Nt);
        for(int i = 0; i < Nt; i++){
            FluxMat[i].resize(dim+1);
            for(int j = 0; j < dim+1; j++){
                FluxMat[i][j].resize(CC.DOF, CC.DOF);
                FluxMat[i][j].setZero();
            }
        }
        Quadrature<2, 14> Quad2D;
        Eigen::MatrixXd Trans = Eigen::MatrixXd::Zero(CC.DOF, CC.DOF);
        Eigen::MatrixXd Alpha = Eigen::MatrixXd::Zero(CC.DOF, CC.DOF);
        for(int i = 0; i < Nt; i++){
            auto curCell = std::dynamic_pointer_cast<SpatialMesh::Cell<dim, dim+1>>(mesh.getCells()[i]);
            Eigen::MatrixXd curVertices = curCell->getVerticesCoordinates();
            Eigen::MatrixXd curPhysicalNodes = phyEquiNodes(dim, CC.POLYDEG, curVertices);
            std::vector<Polynomial::Polynomial> curLB = PolyFem::LagrangeBasis(dim, CC.POLYDEG, curVertices);
            for(int j = 0; j < dim+1; j++){
                int adjCellIndex = curCell->getAdjacentCell(j);
                if (adjCellIndex != -1) {
                    auto adjCell = std::dynamic_pointer_cast<SpatialMesh::Cell<dim, dim+1>>(mesh.getCells()[adjCellIndex]);
                    if (adjCell) {
                        Eigen::MatrixXd adjVertices = adjCell->getVerticesCoordinates();
                        Trans = CordTrans(adjVertices, curPhysicalNodes);
                        for(int k = 0; k < RefBasis.size(); k++){
                            Alpha.row(k) = RefBasis[k].evaluateBatch(Trans).transpose();
                        }
                        FluxMat[i][j] = (Alpha * MassFaceMat[i][j]).transpose();
                    }
                }
                else if(CC.BOUNDARY_COND[curCell->getFaces()[j]->getBoundaryTag()].first == 1){ // Thermalizing boundary condition
                    for(int k = 0; k < CC.DOF; k++){
                        FluxMat[i][j].col(k) = IntFaceMat[i][j];
                    }
                }
                else if(CC.BOUNDARY_COND[curCell->getFaces()[j]->getBoundaryTag()].first == 7){ // Dirichlet boundary condition
                    FluxMat[i][j] = MassFaceMat[i][j];
                    // Eigen::MatrixXd FaceVertices = Eigen::MatrixXd::Zero(dim, dim);
                    // for(int m = 0; m < dim; m++){
                    //     FaceVertices.col(m) = mesh.getCells()[i]->getFaces()[j]->getVertices()[m]->getCoordinates();
                    // }
                    // Eigen::MatrixXd QuadCords = Quad2D.getPoints(FaceVertices);
                    // #pragma omp parallel for
                    // for(int k = 0; k < CC.DOF; k++){
                    //     for(int l = 0; l < CC.DOF; l++){
                    //         Eigen::VectorXd f = curLB[k].evaluateBatch(QuadCords);
                    //         f = f.array() * BoundaryCondition::Analytical(QuadCords, 0.0).array();
                    //         FluxMat[i][j](k, l) = Quad2D.CalQuad(f) * alphaFace * mesh.getCells()[i]->getFaces()[j]->getMeasure();
                    //     }
                    // }
                }
            }
        }
        std::cout << "  >>> FluxMat computed." << std::endl;

        std::cout << ">>> FEM elementwise operations completed." << std::endl;
    }

    template<int dim>
    void Integral<dim>::updateDirichletFlux(const SpatialMesh::SpatialMesh<dim>& mesh, double dt){
        for(int i = 0; i < CC.N_MESH_CELL; i++){
            for(int j = 0; j < dim+1; j++){
                int BCTAG = mesh.getCells()[i]->getFaces()[j]->getBoundaryTag();
                if(BCTAG == 0){
                    continue;
                }
                else if(CC.BOUNDARY_COND[BCTAG].first == 7){
                    FluxMat[i][j] = FluxMat[i][j].array() * exp(dt);
                }
            }
        }
    }

    template<int dim>
    void Integral<dim>::save(std::string results_dir){
        std::cout << ">>> Saving Integral matrices to files..." << std::endl;
        
        // Create results directory if it doesn't exist
        std::filesystem::create_directories(results_dir);
        
        // Save IntMat
        std::ofstream intmat_file(results_dir + "/IntMat.txt");
        if (intmat_file.is_open()) {
            intmat_file << "# Integration Matrix (IntMat)" << std::endl;
            intmat_file << "# Dimensions: " << IntMat.rows() << " x " << IntMat.cols() << std::endl;
            intmat_file << "# Format: Matrix in Eigen format" << std::endl;
            intmat_file << IntMat << std::endl;
            intmat_file.close();
            std::cout << "  >>> IntMat saved to " << results_dir << "/IntMat.txt" << std::endl;
        } else {
            std::cerr << "  >>> Error: Could not open file for IntMat" << std::endl;
        }

        // Save MassMat
        std::ofstream massmat_file(results_dir + "/MassMat.txt");
        if (massmat_file.is_open()) {
            massmat_file << "# Mass Matrix (MassMat)" << std::endl;
            massmat_file << "# Number of elements: " << MassMat.size() << std::endl;
            if (!MassMat.empty()) {
                massmat_file << "# Element matrix dimensions: " << MassMat[0].rows() << " x " << MassMat[0].cols() << std::endl;
            }
            massmat_file << "# Format: Each element matrix separated by '---'" << std::endl;
            for (size_t i = 0; i < MassMat.size(); ++i) {
                massmat_file << "# Element " << i << std::endl;
                massmat_file << MassMat[i] << std::endl;
                if (i < MassMat.size() - 1) {
                    massmat_file << "---" << std::endl;
                }
            }
            massmat_file.close();
            std::cout << "  >>> MassMat saved to " << results_dir << "/MassMat.txt" << std::endl;
        } else {
            std::cerr << "  >>> Error: Could not open file for MassMat" << std::endl;
        }


        // Save IntFaceMat
        std::ofstream intfacemat_file(results_dir + "/IntFaceMat.txt");
        if (intfacemat_file.is_open()) {
            intfacemat_file << "# Integration Face Matrix (IntFaceMat)" << std::endl;
            intfacemat_file << "# Number of elements: " << IntFaceMat.size() << std::endl;
            if (!IntFaceMat.empty() && !IntFaceMat[0].empty()) {
                intfacemat_file << "# Number of faces per element: " << IntFaceMat[0].size() << std::endl;
                intfacemat_file << "# Face vector length: " << IntFaceMat[0][0].size() << std::endl;
            }
            intfacemat_file << "# Format: Element -> Face -> Vector, separated by '---'" << std::endl;
            for (size_t i = 0; i < IntFaceMat.size(); ++i) {
                intfacemat_file << "# Element " << i << std::endl;
                for (size_t j = 0; j < IntFaceMat[i].size(); ++j) {
                    intfacemat_file << "# Face " << j << std::endl;
                    intfacemat_file << IntFaceMat[i][j].transpose() << std::endl;
                    if (j < IntFaceMat[i].size() - 1) {
                        intfacemat_file << "---" << std::endl;
                    }
                }
                if (i < IntFaceMat.size() - 1) {
                    intfacemat_file << "===" << std::endl;
                }
            }
            intfacemat_file.close();
            std::cout << "  >>> IntFaceMat saved to " << results_dir << "/IntFaceMat.txt" << std::endl;
        } else {
            std::cerr << "  >>> Error: Could not open file for IntFaceMat" << std::endl;
        }

        // Save StiffMat
        std::ofstream stiffmat_file(results_dir + "/StiffMat.txt");
        if (stiffmat_file.is_open()) {
            stiffmat_file << "# Stiffness Matrix (StiffMat)" << std::endl;
            stiffmat_file << "# Number of elements: " << StiffMat.size() << std::endl;
            if (!StiffMat.empty() && !StiffMat[0].empty()) {
                stiffmat_file << "# Number of dimensions: " << StiffMat[0].size() << std::endl;
                stiffmat_file << "# Element matrix dimensions: " << StiffMat[0][0].rows() << " x " << StiffMat[0][0].cols() << std::endl;
            }
            stiffmat_file << "# Format: Element -> Dimension -> Matrix, separated by '---'" << std::endl;
            for (size_t i = 0; i < StiffMat.size(); ++i) {
                stiffmat_file << "# Element " << i << std::endl;
                for (size_t j = 0; j < StiffMat[i].size(); ++j) {
                    stiffmat_file << "# Dimension " << j << std::endl;
                    stiffmat_file << StiffMat[i][j] << std::endl;
                    if (j < StiffMat[i].size() - 1) {
                        stiffmat_file << "---" << std::endl;
                    }
                }
                if (i < StiffMat.size() - 1) {
                    stiffmat_file << "===" << std::endl;
                }
            }
            stiffmat_file.close();
            std::cout << "  >>> StiffMat saved to " << results_dir << "/StiffMat.txt" << std::endl;
        } else {
            std::cerr << "  >>> Error: Could not open file for StiffMat" << std::endl;
        }

        
        // Save MassFaceMat
        std::ofstream massfacemat_file(results_dir + "/MassFaceMat.txt");
        if (massfacemat_file.is_open()) {
            massfacemat_file << "# Mass Face Matrix (MassFaceMat)" << std::endl;
            massfacemat_file << "# Number of elements: " << MassFaceMat.size() << std::endl;
            if (!MassFaceMat.empty() && !MassFaceMat[0].empty()) {
                massfacemat_file << "# Number of faces per element: " << MassFaceMat[0].size() << std::endl;
                massfacemat_file << "# Face matrix dimensions: " << MassFaceMat[0][0].rows() << " x " << MassFaceMat[0][0].cols() << std::endl;
            }
            massfacemat_file << "# Format: Element -> Face -> Matrix, separated by '---'" << std::endl;
            for (size_t i = 0; i < MassFaceMat.size(); ++i) {
                massfacemat_file << "# Element " << i << std::endl;
                for (size_t j = 0; j < MassFaceMat[i].size(); ++j) {
                    massfacemat_file << "# Face " << j << std::endl;
                    massfacemat_file << MassFaceMat[i][j] << std::endl;
                    if (j < MassFaceMat[i].size() - 1) {
                        massfacemat_file << "---" << std::endl;
                    }
                }
                if (i < MassFaceMat.size() - 1) {
                    massfacemat_file << "===" << std::endl;
                }
            }
            massfacemat_file.close();
            std::cout << "  >>> MassFaceMat saved to " << results_dir << "/MassFaceMat.txt" << std::endl;
        } else {
            std::cerr << "  >>> Error: Could not open file for MassFaceMat" << std::endl;
        }

        
        // Save FluxMat
        std::ofstream fluxmat_file(results_dir + "/FluxMat.txt");
        if (fluxmat_file.is_open()) {
            fluxmat_file << "# Flux Matrix (FluxMat)" << std::endl;
            fluxmat_file << "# Number of elements: " << FluxMat.size() << std::endl;
            if (!FluxMat.empty() && !FluxMat[0].empty()) {
                fluxmat_file << "# Number of faces per element: " << FluxMat[0].size() << std::endl;
                fluxmat_file << "# Face matrix dimensions: " << FluxMat[0][0].rows() << " x " << FluxMat[0][0].cols() << std::endl;
            }
            fluxmat_file << "# Format: Element -> Face -> Matrix, separated by '---'" << std::endl;
            for (size_t i = 0; i < FluxMat.size(); ++i) {
                fluxmat_file << "# Element " << i << std::endl;
                for (size_t j = 0; j < FluxMat[i].size(); ++j) {
                    fluxmat_file << "# Face " << j << std::endl;
                    fluxmat_file << FluxMat[i][j] << std::endl;
                    if (j < FluxMat[i].size() - 1) {
                        fluxmat_file << "---" << std::endl;
                    }
                }
                if (i < FluxMat.size() - 1) {
                    fluxmat_file << "===" << std::endl;
                }
            }
            fluxmat_file.close();
            std::cout << "  >>> FluxMat saved to " << results_dir << "/FluxMat.txt" << std::endl;
        } else {
            std::cerr << "  >>> Error: Could not open file for FluxMat" << std::endl;
        }
        
        
        // Save metadata summary
        std::ofstream metadata_file(results_dir + "/Integral_metadata.txt");
        if (metadata_file.is_open()) {
            metadata_file << "# Integral Matrix Metadata" << std::endl;
            metadata_file << "# Generated on: " << std::chrono::system_clock::now().time_since_epoch().count() << std::endl;
            metadata_file << std::endl;
            
            metadata_file << "IntMat:" << std::endl;
            metadata_file << "  Dimensions: " << IntMat.rows() << " x " << IntMat.cols() << std::endl;
            metadata_file << "  Size: " << IntMat.size() << " elements" << std::endl;
            metadata_file << std::endl;
            
            metadata_file << "MassMat:" << std::endl;
            metadata_file << "  Number of elements: " << MassMat.size() << std::endl;
            if (!MassMat.empty()) {
                metadata_file << "  Matrix dimensions: " << MassMat[0].rows() << " x " << MassMat[0].cols() << std::endl;
                metadata_file << "  Total size: " << MassMat.size() * MassMat[0].size() << " elements" << std::endl;
            }
            metadata_file << std::endl;
            
            metadata_file << "StiffMat:" << std::endl;
            metadata_file << "  Number of elements: " << StiffMat.size() << std::endl;
            if (!StiffMat.empty() && !StiffMat[0].empty()) {
                metadata_file << "  Dimensions per element: " << StiffMat[0].size() << std::endl;
                metadata_file << "  Matrix dimensions: " << StiffMat[0][0].rows() << " x " << StiffMat[0][0].cols() << std::endl;
                metadata_file << "  Total size: " << StiffMat.size() * StiffMat[0].size() * StiffMat[0][0].size() << " elements" << std::endl;
            }
            metadata_file << std::endl;
            
            metadata_file << "MassFaceMat:" << std::endl;
            metadata_file << "  Number of elements: " << MassFaceMat.size() << std::endl;
            if (!MassFaceMat.empty() && !MassFaceMat[0].empty()) {
                metadata_file << "  Faces per element: " << MassFaceMat[0].size() << std::endl;
                metadata_file << "  Matrix dimensions: " << MassFaceMat[0][0].rows() << " x " << MassFaceMat[0][0].cols() << std::endl;
                metadata_file << "  Total size: " << MassFaceMat.size() * MassFaceMat[0].size() * MassFaceMat[0][0].size() << " elements" << std::endl;
            }
            metadata_file << std::endl;
            
            metadata_file << "FluxMat:" << std::endl;
            metadata_file << "  Number of elements: " << FluxMat.size() << std::endl;
            if (!FluxMat.empty() && !FluxMat[0].empty()) {
                metadata_file << "  Faces per element: " << FluxMat[0].size() << std::endl;
                metadata_file << "  Matrix dimensions: " << FluxMat[0][0].rows() << " x " << FluxMat[0][0].cols() << std::endl;
                metadata_file << "  Total size: " << FluxMat.size() * FluxMat[0].size() * FluxMat[0][0].size() << " elements" << std::endl;
            }
            
            metadata_file.close();
            std::cout << "  >>> Metadata saved to " << results_dir << "/Integral_metadata.txt" << std::endl;
        } else {
            std::cerr << "  >>> Error: Could not open file for metadata" << std::endl;
        }
        
        std::cout << ">>> Integral matrices saved successfully to " << results_dir << "/ directory" << std::endl;
    }

}