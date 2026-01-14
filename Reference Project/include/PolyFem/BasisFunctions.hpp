#pragma once

#include <stdexcept>
#include <algorithm>
#include <fstream>
#include "Utility/math_utils.hpp"
#include "Eigen/Dense"
#include "Polynomial/Polynomial.hpp"
#include <filesystem>
#include <chrono>
#include <iostream>

namespace PolyFem{

    /*
    reference element:
    - 2D: triangle (0,0)-(1,0)-(0,1)
    - 3D: tetrahedron (0,0,0)-(1,0,0)-(0,1,0)-(0,0,1)
    */

    /**
    * @brief Compute equi-distributed nodes on the standard simplex
    * @param order The order of the polynomial
    * @return The nodes (rows: number of dimensions, cols: number of nodes)
    */
    inline Eigen::MatrixXd EquiNodes(int dim, int order){
        int dof, idx;
        Eigen::MatrixXd Nodes;
        if(order < 0){
            throw std::invalid_argument("EquiNodes: Order must be non-negative");
        }
        switch (dim) {
            case 1:
                dof = order + 1;
                Nodes = Eigen::MatrixXd::Zero(1, dof);
                if(order == 0){
                    Nodes(0, 0) = 0.0;
                    break;
                }
                idx = 0;
                for(int i = 0; i <= order; i++){
                    Nodes(0, idx) = i / double(order);
                    idx++;
                }
                break;
            case 2:
                dof = (order + 1) * (order + 2) / 2;
                Nodes = Eigen::MatrixXd::Zero(2, dof);
                if(order == 0){
                    Nodes.setZero();
                    break;
                }
                idx = 0;
                for(int i = 0; i <= order; i++){
                    for(int j = 0; j <= order - i; j++){
                        Nodes(0, idx) = j / double(order);
                        Nodes(1, idx) = i / double(order);
                        idx++;
                    }
                }
                break;
            case 3:
                dof = (order + 1) * (order + 2) * (order + 3) / 6;
                Nodes = Eigen::MatrixXd::Zero(3, dof);
                if(order == 0){
                    Nodes.setZero();
                    break;
                }
                idx = 0;
                for(int i = 0; i <= order; i++){
                    for(int j = 0; j <= order - i; j++){
                        for(int k = 0; k <= order - i - j; k++){
                            Nodes(0, idx) = k / double(order);
                            Nodes(1, idx) = j / double(order);
                            Nodes(2, idx) = i / double(order);
                            idx++;
                        }
                    }
                }
                break;
            default:
                throw std::invalid_argument("EquiNodes: Dimension must be 1, 2 or 3");
        }
        return Nodes;
    }

    /**
    * @brief Compute the barycentric coordinates of a point in the reference element
    * @param dim The dimension of the reference element
    * @param Cartesian The Cartesian coordinates of the point (rows: dimension, cols: number of points)
    * @return The barycentric coordinates of the point
    * L1 = 1 - L2 - L3 (-L4), L2 = x, L3 = y (L4 = z)
    */
    inline Eigen::MatrixXd Barycentric(int dim, Eigen::MatrixXd Cartesian) {
        Eigen::MatrixXd barycentric = Eigen::MatrixXd::Zero(dim + 1, Cartesian.cols());
        switch (dim) {
            case 1:
                barycentric.row(1) = Cartesian.row(0);
                barycentric.row(0) = 1.0 - Cartesian.row(0).array();
                break;
            case 2:
                barycentric.row(1) = Cartesian.row(0);
                barycentric.row(2) = Cartesian.row(1);
                barycentric.row(0) = 1.0 - Cartesian.row(0).array() - Cartesian.row(1).array();
                break;
            case 3:
                barycentric.row(1) = Cartesian.row(0);
                barycentric.row(2) = Cartesian.row(1);
                barycentric.row(3) = Cartesian.row(2);
                barycentric.row(0) = 1.0 - Cartesian.row(0).array() - Cartesian.row(1).array() - Cartesian.row(2).array();
                break;
            default:
                throw std::invalid_argument("Barycentric: Dimension must be 1, 2 or 3");
        }
        return barycentric;
    }

    /**
    * @brief Compute the Cartesian coordinates of a point in the reference element
    * @param dim The dimension of the reference element
    * @param barycentric The barycentric coordinates of the point (rows: dimension, cols: number of points)
    * @return The Cartesian coordinates of the point
    * x = L2, y = L3 (z = L4)
    */
    inline Eigen::MatrixXd Cartesian(int dim, Eigen::MatrixXd barycentric) {
        Eigen::MatrixXd cartesian = Eigen::MatrixXd::Zero(dim, barycentric.cols());
        switch (dim) {
            case 1:
                cartesian.row(0) = barycentric.row(1).array();
                break;
            case 2:
                cartesian.row(0) = barycentric.row(1).array();
                cartesian.row(1) = barycentric.row(2).array();
                break;
            case 3:
                cartesian.row(0) = barycentric.row(1).array();
                cartesian.row(1) = barycentric.row(2).array();
                cartesian.row(2) = barycentric.row(3).array();
                break;
            default:
                throw std::invalid_argument("Cartesian: Dimension must be 1, 2 or 3");
        }
        return cartesian;
    }

    /**
    * @brief Compute the equi-distributed nodes on the physical simplex
    * @param dim The dimension of the physical simplex
    * @param order The order of the polynomial
    * @param Vertices The vertices of the physical simplex
    * @return The equi-distributed nodes on the physical simplex
    */
    inline Eigen::MatrixXd phyEquiNodes(int dim, int order, Eigen::MatrixXd Vertices){
        if(dim != Vertices.rows() || dim + 1 != Vertices.cols()){
            throw std::invalid_argument("phyEquiNodes: The number or dimension of vertices is incorrect.");
        }
        int dof;
        switch (dim) {
            case 1:
                dof = order + 1;
                break;
            case 2:
                dof = (order + 1) * (order + 2) / 2;
                break;
            case 3:
                dof = (order + 1) * (order + 2) * (order + 3) / 6;
                break;
            default:
                throw std::invalid_argument("phyEquiNodes: Dimension must be 1, 2 or 3");
        }
        Eigen::MatrixXd RefNodes = EquiNodes(dim, order);
        Eigen::MatrixXd refBarycentric = Barycentric(dim, RefNodes);
        Eigen::MatrixXd Nodes = Eigen::MatrixXd::Zero(dim, dof);
        for(int i = 0; i < refBarycentric.cols(); i++){
            Nodes.col(i) = Vertices * refBarycentric.col(i);
        }
        return Nodes;
    }

    /**
    * @brief Build affine coefficient matrix that maps physical Cartesian coordinates to reference Cartesian coordinates
    *        on a simplex. For 2D, A is 2x3 such that [r;s] = A * [1; x; y]. For 3D, A is 3x4 such that [r;s;t] = A * [1; x; y; z].
    */
    inline Eigen::MatrixXd mapping(const Eigen::MatrixXd& Ver) {
        if (Ver.rows() == 2 && Ver.cols() == 3) {
            // 2x3 case: solve for a11, a12, a13, a21, a22, a23
            double x1 = Ver(0, 0), y1 = Ver(1, 0);
            double x2 = Ver(0, 1), y2 = Ver(1, 1);
            double x3 = Ver(0, 2), y3 = Ver(1, 2);

            double Delta = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1);
            if (std::abs(Delta) < 1e-16) {
                throw std::runtime_error("Delta is zero, no unique solution.");
            }

            Eigen::MatrixXd A(2, 3);
            A(0, 0) = (y1 * x3 - x1 * y3) / Delta; // a11
            A(0, 1) = (y3 - y1) / Delta;           // a12
            A(0, 2) = (x1 - x3) / Delta;           // a13
            A(1, 0) = (x1 * y2 - y1 * x2) / Delta; // a21
            A(1, 1) = (y1 - y2) / Delta;           // a22
            A(1, 2) = (x2 - x1) / Delta;           // a23

            return A;
        } else if (Ver.rows() == 3 && Ver.cols() == 4) {
            // 3x4 case: solve for a11, ..., a34
            double x1 = Ver(0, 0), y1 = Ver(1, 0), z1 = Ver(2, 0);
            double x2 = Ver(0, 1), y2 = Ver(1, 1), z2 = Ver(2, 1);
            double x3 = Ver(0, 2), y3 = Ver(1, 2), z3 = Ver(2, 2);
            double x4 = Ver(0, 3), y4 = Ver(1, 3), z4 = Ver(2, 3);

            // Compute Delta
            double Delta = 
                (x2 - x1) * ((y3 - y1) * (z4 - z1) - (y4 - y1) * (z3 - z1)) -
                (y2 - y1) * ((x3 - x1) * (z4 - z1) - (x4 - x1) * (z3 - z1)) +
                (z2 - z1) * ((x3 - x1) * (y4 - y1) - (x4 - x1) * (y3 - y1));

            if (std::abs(Delta) < 1e-24) {
                throw std::runtime_error("Delta is zero, no unique solution.");
            }

            // Compute a_{i2}, a_{i3}, a_{i4} for i=1,2,3
            Eigen::MatrixXd A(3, 4);

            // a_{12}, a_{13}, a_{14}
            A(0, 1) = ((y3 - y1) * (z4 - z1) - (y4 - y1) * (z3 - z1)) / Delta;  // a12
            A(0, 2) = -((x3 - x1) * (z4 - z1) - (x4 - x1) * (z3 - z1)) / Delta; // a13
            A(0, 3) = ((x3 - x1) * (y4 - y1) - (x4 - x1) * (y3 - y1)) / Delta;  // a14

            // a_{22}, a_{23}, a_{24}
            A(1, 1) = -((y2 - y1) * (z4 - z1) - (y4 - y1) * (z2 - z1)) / Delta; // a22
            A(1, 2) = ((x2 - x1) * (z4 - z1) - (x4 - x1) * (z2 - z1)) / Delta;  // a23
            A(1, 3) = -((x2 - x1) * (y4 - y1) - (x4 - x1) * (y2 - y1)) / Delta; // a24

            // a_{32}, a_{33}, a_{34}
            A(2, 1) = ((y2 - y1) * (z3 - z1) - (y3 - y1) * (z2 - z1)) / Delta;  // a32
            A(2, 2) = -((x2 - x1) * (z3 - z1) - (x3 - x1) * (z2 - z1)) / Delta; // a33
            A(2, 3) = ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / Delta;  // a34

            // Compute a_{i1} = -x1 * a_{i2} - y1 * a_{i3} - z1 * a_{i4}
            for (int i = 0; i < 3; ++i) {
                A(i, 0) = -x1 * A(i, 1) - y1 * A(i, 2) - z1 * A(i, 3);
            }

            return A;
        } else {
            throw std::invalid_argument("Ver must be either 2x3 or 3x4.");
        }
    }

    /**
    * @brief Transform physical Cartesian coordinates to reference Cartesian coordinates on the simplex.
    *        For triangles, returns [r;s] where r=L2, s=L3. For tetrahedra, returns [r;s;t] where r=L2, s=L3, t=L4.
    * @param Vertices row: coordinates, col: vertices (physical simplex vertices)
    * @param Points row: coordinates, col: points (physical Cartesian coordinates)
    */
    inline Eigen::MatrixXd CordTrans(Eigen::MatrixXd Vertices, Eigen::MatrixXd Points){
        if(Vertices.rows() != Points.rows()){
            throw std::invalid_argument("Dimension of vertices and points must match");
        }

        Eigen::MatrixXd A = mapping(Vertices);
        Eigen::MatrixXd C = Eigen::MatrixXd::Ones(Points.rows() + 1, Points.cols());
        C.block(1, 0, Points.rows(), Points.cols()) = Points;
        Eigen::MatrixXd B = A * C;
        return B; 
    }

    /**
    * @brief Compute the derivative of the linear mapping from the reference simplex to the physical simplex
    * @param dim The dimension of the physical simplex
    * @param Vertices The vertices of the physical simplex (col: vertices, row: coordinates)
    * @return The derivative of the linear mapping from the reference simplex to the physical simplex
    * [[x_r x_s x_t]
    *  [y_r y_s y_t]
    *  [z_r z_s z_t]]
    */
    inline Eigen::MatrixXd GradXr(int dim, Eigen::MatrixXd Vertices) {
        if(Vertices.rows() != dim || Vertices.cols() != dim + 1) {
            throw std::invalid_argument("Dimension of vertices must match the dimension of the basis");
        }

        Eigen::MatrixXd pdXr = Eigen::MatrixXd::Zero(dim, dim);
        switch (dim)
        {
            case 1:
                pdXr(0, 0) = Vertices(0, 1) - Vertices(0, 0);
                break;
            case 2:
                pdXr.col(0) = Vertices.col(1) - Vertices.col(0);
                pdXr.col(1) = Vertices.col(2) - Vertices.col(0);
                break;
            case 3:
                pdXr.col(0) = Vertices.col(1) - Vertices.col(0);
                pdXr.col(1) = Vertices.col(2) - Vertices.col(0);
                pdXr.col(2) = Vertices.col(3) - Vertices.col(0);
                break;
            default:
                throw std::invalid_argument("GradXr: Dimension must be 1, 2 or 3");
                break;
        }
        return pdXr;
    }

    /**
    * @brief Compute the derivative of the linear mapping from the physical simplex to the reference simplex
    * @param dim The dimension of the simplex
    * @param Vertices The vertices of the physical simplex (col: vertices, row: coordinates)
    * @return The derivative of the linear mapping from the physical simplex to the reference simplex
    * [[r_x r_y r_z]
    *  [s_x s_y s_z]
    *  [t_x t_y t_z]]
    */
    inline Eigen::MatrixXd GradRx(int dim, Eigen::MatrixXd Vertices) {
        return GradXr(dim, Vertices).inverse();
    }

    /**
    * @brief Compute the Jacobian (determinant) of the linear mapping from the reference simplex to the physical simplex
    * @param dim The dimension of the physical simplex
    * @param Vertices The vertices of the physical simplex (col: vertices, row: coordinates)
    * @return The Jacobian determinant of the linear mapping from the reference simplex to the physical simplex
    */
    inline double Jacobian(int dim, Eigen::MatrixXd Vertices) {
        Eigen::MatrixXd gradXr = GradXr(dim, Vertices);
        if(gradXr.rows() != dim || gradXr.cols() != dim) {
            throw std::invalid_argument("Dimension of gradXr must match the dimension");
        }
        return gradXr.determinant();
    }

    /**
    * @brief Compute Lagrangian basis functions
    * @param deg The degree of the polynomial
    * @param NodalDis The nodal distribution (rows: number of dimensions, cols: number of nodes)
    * @return The Lagrangian basis functions (a polynomial is represented by a row vector)
    */
    inline Eigen::MatrixXd LagrangianBasis(int dim, int order, Eigen::MatrixXd NodalDis){
        int dof;
        switch (dim) {
            case 1:
                dof = order + 1;
                break;
            case 2:
                dof = (order + 1) * (order + 2) / 2;
                break;
            case 3:
                dof = (order + 1) * (order + 2) * (order + 3) / 6;
                break;
            default:
                throw std::invalid_argument("LagrangianBasis: Dimension must be 1, 2 or 3");
        }
        if(dof != NodalDis.cols() || dim != NodalDis.rows()){
            throw std::invalid_argument("LagrangianBasis: The number of nodes isn't consistent with the degree or the dimension.");
        }
        Eigen::MatrixXd Coef = Eigen::MatrixXd::Zero(dof, dof); // each row corresponds to the coefficients of a single LP
        Eigen::MatrixXd M = Eigen::MatrixXd::Zero(dof, dof);
        Eigen::VectorXd b = Eigen::VectorXd::Zero(dof);
        int idx = 0;
        switch (dim) {
            case 1:
                for(int i = 0; i <= order; i++){
                    M.col(idx) = NodalDis.row(0).array().pow(i);
                    idx++;
                }
                for(int i = 0; i < dof; i++){
                    b.setZero();
                    b(i) = 1.0;
                    Coef.row(i) = M.lu().solve(b);
                }
                break;
            case 2:
                for(int d = 0; d <= order; d++){
                    for(int dy = 0; dy <= d; dy++){
                        int dx = d - dy;
                        M.col(idx) = NodalDis.row(0).array().pow(dx) * NodalDis.row(1).array().pow(dy);
                        idx ++;
                    }
                }
                for(int i = 0; i < dof; i++){
                    b.setZero();
                    b(i) = 1.0;
                    Coef.row(i) = M.lu().solve(b);
                }
                break;
            case 3:
                for(int d = 0; d <= order; d++){
                    for(int dx = d; dx >= 0; dx--){
                        for(int dy = d-dx; dy >= 0; dy--){
                            int dz = d - dx - dy;
                            M.col(idx) = NodalDis.row(0).array().pow(dx) * NodalDis.row(1).array().pow(dy) * NodalDis.row(2).array().pow(dz);
                            idx ++;
                        }
                    }
                }
                for(int i = 0; i < dof; i++){
                    b.setZero();
                    b(i) = 1.0;
                    Coef.row(i) = M.lu().solve(b);
                }
                break;
            default:
                throw std::invalid_argument("LagrangianBasis: Dimension must be 2 or 3");
        }

        for(int a = 0; a < Coef.rows(); a++){
            for(int b = 0; b < Coef.cols(); b++){
                if(std::isnan(Coef(a, b))){
                    throw std::runtime_error("LagrangianBasis: Coefficient is nan");
                }
            }
        }

        return Coef;
    }

    /**
    * @brief Compute the standard basis functions on the reference simplex
    * @param dim The dimension of the reference simplex
    * @param order The order of the polynomial
    * @return The standard basis functions
    */
    inline std::vector<Polynomial::Polynomial> ReferenceBasis(int dim, int order) {
        Eigen::MatrixXd IntPts = EquiNodes(dim, order);
        std::vector<Polynomial::Polynomial> basis;
        int dof;
        Eigen::MatrixXd Coef;
        switch (dim) {
            case 1:
                dof = order + 1;
                Coef = LagrangianBasis(1, order, IntPts);
                for(int i = 0; i < dof; i++){
                    basis.push_back(Polynomial::Polynomial(1, Coef.row(i)));
                }
                break;
            case 2:
                dof = (order + 1) * (order + 2) / 2;
                Coef = LagrangianBasis(2, order, IntPts);
                for(int i = 0; i < dof; i++){
                    basis.push_back(Polynomial::Polynomial(2, Coef.row(i)));
                }
                break;
            case 3:
                dof = (order + 1) * (order + 2) * (order + 3) / 6;
                Coef = LagrangianBasis(3, order, IntPts);
                for(int i = 0; i < dof; i++){
                    basis.push_back(Polynomial::Polynomial(3, Coef.row(i)));
                }
                break;
            default:
                throw std::invalid_argument("ReferenceBasis: Dimension must be 1, 2 or 3");
        }
        return basis;
    }

    inline std::vector<Polynomial::Polynomial> ModalBasis(int dim, int dof) {
        std::vector<Polynomial::Polynomial> basis;
        for(int i = 0; i < dof; i++){
            Eigen::VectorXd Coef = Eigen::VectorXd::Zero(dof);
            Coef(i) = 1.0;
            basis.push_back(Polynomial::Polynomial(dim, Coef));
        }
        return basis;
    }

    inline std::vector<Polynomial::Polynomial> LagrangeBasis(int dim, int order, Eigen::MatrixXd Vertices) {
        Eigen::MatrixXd IntPts = phyEquiNodes(dim, order, Vertices);
        std::vector<Polynomial::Polynomial> basis;
        int dof;
        Eigen::MatrixXd Coef;
        switch (dim) {
            case 1:
                dof = order + 1;
                Coef = LagrangianBasis(1, order, IntPts);
                for(int i = 0; i < dof; i++){
                    basis.push_back(Polynomial::Polynomial(1, Coef.row(i)));
                }
                break;
            case 2:
                dof = (order + 1) * (order + 2) / 2;
                Coef = LagrangianBasis(2, order, IntPts);
                for(int i = 0; i < dof; i++){
                    basis.push_back(Polynomial::Polynomial(2, Coef.row(i)));
                }
                break;
            case 3:
                dof = (order + 1) * (order + 2) * (order + 3) / 6;
                Coef = LagrangianBasis(3, order, IntPts);
                for(int i = 0; i < dof; i++){
                    basis.push_back(Polynomial::Polynomial(3, Coef.row(i)));
                }
                break;
            default:
                throw std::invalid_argument("ReferenceBasis: Dimension must be 1, 2 or 3");
        }
        return basis;
    }

}