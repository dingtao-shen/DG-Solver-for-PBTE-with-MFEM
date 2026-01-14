#pragma once

#include <Eigen/Dense>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <memory>
#include <vector>
#include <iostream>
#include <limits>
#include <cmath>
#include <algorithm>

namespace SpatialMesh {

    template <int dim> class Node;
    template <int dim, int verticesCount> class Face;
    template <int dim, int verticesCount> class Cell;

    template <typename T>
    using Ptr = std::shared_ptr<T>;

    using Node2DPtr = Ptr<Node<2>>;
    using Node3DPtr = Ptr<Node<3>>;

    using Face2DPtr = Ptr<Face<2, 2>>;
    using Face3DPtr = Ptr<Face<3, 3>>;

    using Cell2DPtr = Ptr<Cell<2, 3>>; 
    using Cell3DPtr = Ptr<Cell<3, 4>>; 

    /********************************* base node class *********************************/

    template <int dim>
    class Node {
        protected:
            Eigen::VectorXd coordinates;
            int index;
        public:
            // constructor
            Node() {
                coordinates.resize(dim);
                coordinates.setZero();
                index = -1;
            }
            
            explicit Node(const Eigen::VectorXd& coords, int idx = -1) : coordinates(coords), index(idx) {
                if (coords.size() != dim) {
                    throw std::invalid_argument("Coordinate dimension mismatch");
                }
            }
            
            // dimension specific constructor
            template <int D = dim, typename = std::enable_if_t<D == 2>>
            Node(double x, double y, int idx = -1) : index(idx) {
                coordinates.resize(2);
                coordinates << x, y;
            }

            template <int D = dim, typename = std::enable_if_t<D == 3>>
            Node(double x, double y, double z, int idx = -1) : index(idx) {
                coordinates.resize(3);
                coordinates << x, y, z;
            }

            explicit Node(std::initializer_list<double> list, int idx = -1) {
                if (static_cast<int>(list.size()) != dim) {
                    throw std::invalid_argument("Coordinate dimension mismatch");
                }
                int i = 0;
                coordinates.resize(dim);
                for (double v : list) coordinates(i++) = v;
                index = idx;
            }

            virtual ~Node() = default;

            // setters and getters
            int getDimension() const { return dim; }

            int getIndex() const { return index; }
            const Eigen::VectorXd& getCoordinates() const { return coordinates; }

            template <int D = dim, typename = std::enable_if_t<D == 1 || D == 2 || D == 3>>
            double x() const { return coordinates(0); }

            template <int D = dim, typename = std::enable_if_t<D == 2 || D == 3>>
            double y() const { return coordinates(1); }

            template <int D = dim, typename = std::enable_if_t<D == 3>>
            double z() const { return coordinates(2); }    

            bool operator==(const Node<dim>& other) const {
                if (index != other.index) return false;
                return coordinates.isApprox(other.coordinates);
            }
    
            bool operator!=(const Node<dim>& other) const {
                return !(*this == other);
            }
    };

    /********************************* base face class *********************************/
    
    template <int dim, int verticesCount>
    class Face {
        protected:
            using TypedNodePtr = std::conditional_t<dim == 2, Node2DPtr, Node3DPtr>;
            std::vector<TypedNodePtr> vertices;
            int index = -1;
            int boundary_tag = 0;
            int periodic_pair = -1;
            std::vector<int> adjacent_cells;
            std::vector<int> local_indices;

            double measure = -1.0;
            Eigen::VectorXd normal;

        public:
            Face() {
                static_assert((dim == 2 && verticesCount == 2) || 
                        (dim == 3 && verticesCount == 3),
                        "Invalid vertices count for dimension");
                vertices.resize(verticesCount);
                adjacent_cells.resize(2, -1);
                local_indices.resize(2, -1);
            }

            explicit Face(const std::vector<TypedNodePtr>& verts, int idx = -1, int bd_tag = 0)
            {
                if (verts.size() != verticesCount) {
                    throw std::invalid_argument("Invalid number of vertices");
                }

                vertices.resize(verticesCount);
                vertices = verts;
                index = idx;
                boundary_tag = bd_tag;
                adjacent_cells = std::vector<int>(2, -1);
                local_indices = std::vector<int>(2, -1);

                if constexpr (dim == 2) {
                    const auto& v0 = *std::static_pointer_cast<Node<2>>(vertices[0]);
                    const auto& v1 = *std::static_pointer_cast<Node<2>>(vertices[1]);
                    measure = (v1.getCoordinates() - v0.getCoordinates()).norm();
                    Eigen::Vector2d dir = v1.getCoordinates() - v0.getCoordinates();
                    normal = Eigen::VectorXd(Eigen::Vector2d(dir.y(), -dir.x()).normalized());
                } else if constexpr (dim == 3) {
                    const auto& v0 = *std::static_pointer_cast<Node<3>>(vertices[0]);
                    const auto& v1 = *std::static_pointer_cast<Node<3>>(vertices[1]);
                    const auto& v2 = *std::static_pointer_cast<Node<3>>(vertices[2]);
                    Eigen::Vector3d v1_coords = v1.getCoordinates() - v0.getCoordinates();
                    Eigen::Vector3d v2_coords = v2.getCoordinates() - v0.getCoordinates();
                    measure = v1_coords.cross(v2_coords).norm() / 2.0;
                    normal = Eigen::VectorXd(v1_coords.cross(v2_coords).normalized());
                }
            }

            virtual ~Face() = default;

            // setters and getters
            int getDimension() const { return dim; }
            int getIndex() const { return index; }
            int getBoundaryTag() const { return boundary_tag; }
            int getPeriodicPair() const { return periodic_pair; }
            void setPeriodicPair(int pair) { periodic_pair = pair; }
            bool isBoundary() const { return boundary_tag > 0; }
            bool isPeriodic() const { return periodic_pair != -1; }
            int getNumVertices() const { return vertices.size(); }
            const std::vector<TypedNodePtr>& getVertices() const { return vertices; }
            const TypedNodePtr& getVertex(int i) const {
                if (i < 0 || i >= vertices.size()) {
                    throw std::out_of_range("Vertex index out of range");
                }
                return vertices[i];
            }
            double getMeasure() const { return measure; }
            const Eigen::VectorXd& getNormal() const { return normal; }
            const std::vector<int>& getAdjacentCells() const { return adjacent_cells; }
            const std::vector<int>& getLocalIndices() const { return local_indices; }
            void setAdjacentCells(const std::vector<int>& cells) {
                if (cells.size() != 2) {
                    throw std::invalid_argument("Number of adjacent cells must be equal to 2.");
                }
                adjacent_cells = cells;
            }
            void setAdjacentCell(int i, int cell) {
                if (i < 0 || i >= 2) {
                    throw std::out_of_range("Adjacent cell index out of range");
                }
                adjacent_cells[i] = cell;
            }
            void setLocalIndices(const std::vector<int>& indices) {
                if (indices.size() != 2) {
                    throw std::invalid_argument("Number of local indices must be equal to 2.");
                }
                local_indices = indices;
            }
            void setLocalIndex(int i, int local_index) {
                if (i < 0 || i >= 2) {
                    throw std::out_of_range("Local index out of range");
                }
                local_indices[i] = local_index;
            }
            
            // operators
            bool hasVertex(const TypedNodePtr& node) const {
                if (!node) return false;
                for (const auto& vertex : vertices) {
                    if (vertex && *vertex == *node) {
                        return true;
                    }
                }
                return false;
            }

            bool isPointOnFace(const TypedNodePtr& point, double tolerance = 1e-12) const {
                if constexpr (dim == 2) {
                    const auto& v0 = *std::static_pointer_cast<Node<2>>(vertices[0]);
                    const auto& v1 = *std::static_pointer_cast<Node<2>>(vertices[1]);
                    const auto& p = *std::static_pointer_cast<Node<2>>(point);
                    
                    Eigen::Vector2d v1_coords = v1.getCoordinates() - v0.getCoordinates();
                    Eigen::Vector2d v2_coords = p.getCoordinates() - v0.getCoordinates();
                    double cross = v1_coords(0) * v2_coords(1) - v1_coords(1) * v2_coords(0);
                    double len2 = v1_coords.squaredNorm();
                    if (len2 < std::numeric_limits<double>::epsilon()) return false; // degenerate edge
                    if (std::abs(cross) > tolerance * len2) return false;
    
                    double dot = v1_coords.dot(v2_coords);
                    return dot >= -tolerance * len2 && dot <= len2 + tolerance * len2;
                }
                else if constexpr (dim == 3) {
                    const auto& v0 = *std::static_pointer_cast<Node<3>>(vertices[0]);
                    const auto& v1 = *std::static_pointer_cast<Node<3>>(vertices[1]);
                    const auto& v2 = *std::static_pointer_cast<Node<3>>(vertices[2]);
                    const auto& p = *std::static_pointer_cast<Node<3>>(point);
                    
                    Eigen::Vector3d v1_coords = v1.getCoordinates() - v0.getCoordinates();
                    Eigen::Vector3d v2_coords = v2.getCoordinates() - v0.getCoordinates();
                    Eigen::Vector3d v3_coords = p.getCoordinates() - v0.getCoordinates();

                    Eigen::Vector3d n = v1_coords.cross(v2_coords);
                    double n_norm = n.norm();
                    if (n_norm < std::numeric_limits<double>::epsilon()) return false; // degenerate face

                    double plane_distance = std::abs(n.dot(v3_coords)) / n_norm;
                    double max_edge = std::max({v1.norm(), v2.norm(), (v2 - v1).norm()});
                    if (plane_distance > tolerance * max_edge) return false;

                    // Compute barycentric coordinates relative to triangle (v0, v1, v2)
                    // Using: beta = (w x v)·n / |n|^2, gamma = (u x w)·n / |n|^2, alpha = 1 - beta - gamma
                    // where u=v1, v=v2, w=v3
                    const double denom = n.squaredNorm();
                    const double beta  = (v3_coords.cross(v2_coords)).dot(n) / denom; // L2
                    const double gamma = (v1_coords.cross(v3_coords)).dot(n) / denom; // L3
                    const double alpha = 1.0 - beta - gamma;                            // L1

                    return (alpha >= -tolerance) && (beta >= -tolerance) && (gamma >= -tolerance);
                }
                else{
                    throw std::runtime_error("Can't check if point is on face for dynamic dimension face.");
                }
            }

            bool isPointOnFace(const Eigen::VectorXd& point, double tolerance = 1e-12) const {
                if constexpr (dim == 2) {
                    const auto& v0 = *std::static_pointer_cast<Node<2>>(vertices[0]);
                    const auto& v1 = *std::static_pointer_cast<Node<2>>(vertices[1]);
                    
                    Eigen::Vector2d v1_coords = v1.getCoordinates() - v0.getCoordinates();
                    Eigen::Vector2d v2_coords = point - v0.getCoordinates();
                    double cross = v1_coords(0) * v2_coords(1) - v1_coords(1) * v2_coords(0);
                    double len2 = v1_coords.squaredNorm();
                    if (len2 < std::numeric_limits<double>::epsilon()) return false; // degenerate edge
                    if (std::abs(cross) > tolerance * len2) return false;
    
                    double dot = v1_coords.dot(v2_coords);
                    return dot >= -tolerance * len2 && dot <= len2 + tolerance * len2;
                }
                else if constexpr (dim == 3) {
                    const auto& v0 = *std::static_pointer_cast<Node<3>>(vertices[0]);
                    const auto& v1 = *std::static_pointer_cast<Node<3>>(vertices[1]);
                    const auto& v2 = *std::static_pointer_cast<Node<3>>(vertices[2]);
                    
                    Eigen::Vector3d v1_coords = v1.getCoordinates() - v0.getCoordinates();
                    Eigen::Vector3d v2_coords = v2.getCoordinates() - v0.getCoordinates();
                    Eigen::Vector3d v3_coords = point - v0.getCoordinates();

                    Eigen::Vector3d n = v1_coords.cross(v2_coords);
                    double n_norm = n.norm();
                    if (n_norm < std::numeric_limits<double>::epsilon()) return false; // degenerate face

                    double plane_distance = std::abs(n.dot(v3_coords)) / n_norm;
                    double max_edge = std::max({v1.norm(), v2.norm(), (v2 - v1).norm()});
                    if (plane_distance > tolerance * max_edge) return false;

                    // Compute barycentric coordinates relative to triangle (v0, v1, v2)
                    const double denom = n.squaredNorm();
                    const double beta  = (v3_coords.cross(v2_coords)).dot(n) / denom; // L2
                    const double gamma = (v1_coords.cross(v3_coords)).dot(n) / denom; // L3
                    const double alpha = 1.0 - beta - gamma;                            // L1

                    return (alpha >= -tolerance) && (beta >= -tolerance) && (gamma >= -tolerance);
                }
                else{
                    throw std::runtime_error("Can't check if point is on face for dynamic dimension face.");
                }
            }
    
            bool operator==(const Face& other) const {
                return vertices == other.vertices && index == other.index && boundary_tag == other.boundary_tag;
            }
    
            bool operator!=(const Face& other) const {
                return !(*this == other);
            }

    };

    /***************************** base cell class *********************************/

    template <int dim, int verticesCount>
    class Cell {
        protected:
            using TypedNodePtr = std::conditional_t<dim == 2, Node2DPtr, Node3DPtr>;
            using TypedFacePtr = std::conditional_t<dim == 2, Face2DPtr, Face3DPtr>;
            std::vector<TypedNodePtr> vertices;
            std::vector<TypedFacePtr> faces;
            std::vector<int> adjacent_cells;
            int index = -1;
            Eigen::MatrixXd outward_norm_vec; // each column is a normal vector w.r.t. a face
            double measure = -1.0;
            Node<dim> centroid;
            Eigen::MatrixXd JacobianMatrix;
        
        public:
            Cell() {
                vertices.reserve(verticesCount);
                faces.reserve(dim + 1);
                adjacent_cells.resize(dim + 1, -1);
                outward_norm_vec.resize(dim, dim + 1);
            }

            explicit Cell(const std::vector<TypedNodePtr>& verts, int idx = -1) : vertices(verts), index(idx) {
                if (verts.size() != verticesCount) {
                    throw std::invalid_argument("Invalid number of vertices");
                }

                faces.reserve(dim + 1);
                adjacent_cells.resize(dim + 1, -1);
                outward_norm_vec.resize(dim, dim + 1);
                
                for(int i = 0; i < dim + 1; i++){
                    std::vector<TypedNodePtr> face_vertices;
                    face_vertices.reserve(dim);
                    for(int j = 0; j < dim; j++){
                        face_vertices.push_back(vertices[(i + j) % (dim + 1)]);
                    }
                    faces.push_back(std::make_shared<Face<dim, dim>>(face_vertices));
                }

                // compute measure
                if constexpr (dim == 2 && verticesCount == 3) {
                    const auto& n0 = *std::static_pointer_cast<Node<2>>(vertices[0]);
                    const auto& n1 = *std::static_pointer_cast<Node<2>>(vertices[1]);
                    const auto& n2 = *std::static_pointer_cast<Node<2>>(vertices[2]);

                    Eigen::Vector2d v1 = n1.getCoordinates() - n0.getCoordinates();
                    Eigen::Vector2d v2 = n2.getCoordinates() - n0.getCoordinates();
                    measure = std::abs(v1.x() * v2.y() - v1.y() * v2.x()) / 2.0;
                    JacobianMatrix.resize(2, 2);
                    JacobianMatrix.col(0) = v1;
                    JacobianMatrix.col(1) = v2;
                    if (abs(abs(JacobianMatrix.determinant()) - measure * 2.0) > 1.0e-12) {
                        std::cout << "Jacobian matrix determinant: " << JacobianMatrix.determinant() << std::endl;
                        std::cout << "Measure: " << measure << std::endl;
                        throw std::runtime_error("Jacobian matrix determinant doesn't match the measure");
                    };
                } else if constexpr (dim == 3 && verticesCount == 4) {
                    const auto& n0 = *std::static_pointer_cast<Node<3>>(vertices[0]);
                    const auto& n1 = *std::static_pointer_cast<Node<3>>(vertices[1]);
                    const auto& n2 = *std::static_pointer_cast<Node<3>>(vertices[2]);
                    const auto& n3 = *std::static_pointer_cast<Node<3>>(vertices[3]);

                    Eigen::Vector3d v1 = n1.getCoordinates() - n0.getCoordinates();
                    Eigen::Vector3d v2 = n2.getCoordinates() - n0.getCoordinates();
                    Eigen::Vector3d v3 = n3.getCoordinates() - n0.getCoordinates();
                    measure = std::abs(v1.cross(v2).dot(v3)) / 6.0;
                    JacobianMatrix.resize(3, 3);
                    JacobianMatrix.col(0) = v1;
                    JacobianMatrix.col(1) = v2;
                    JacobianMatrix.col(2) = v3;
                    if (abs(abs(JacobianMatrix.determinant()) - measure * 6.0) > 1.0e-12) {
                        std::cout << "Jacobian matrix determinant: " << JacobianMatrix.determinant() << std::endl;
                        std::cout << "Measure: " << measure << std::endl;
                        throw std::runtime_error("Jacobian matrix determinant doesn't match the measure");
                    };
                }

                // compute centroid
                Eigen::VectorXd cached_centroid = Eigen::VectorXd::Zero(dim);
                for (const auto& vertex : vertices) {
                    cached_centroid += std::static_pointer_cast<Node<dim>>(vertex)->getCoordinates();
                }
                cached_centroid /= vertices.size();
                centroid = Node<dim>(cached_centroid, -1);

                // compute outward normal vectors
                for (int i = 0; i < dim + 1; ++i) {
                    Eigen::VectorXd face_normal = faces[i]->getNormal();
                    Eigen::VectorXd face_centroid = Eigen::VectorXd::Zero(dim);

                    const auto& face = *std::static_pointer_cast<Face<dim, dim>>(faces[i]);
                    for (const auto& v : face.getVertices()) {
                        face_centroid += std::static_pointer_cast<Node<dim>>(v)->getCoordinates();
                    }
                    face_centroid /= dim;
                    if (face_normal.dot(face_centroid - centroid.getCoordinates()) < 0) {
                        face_normal = -face_normal;
                    }
                    outward_norm_vec.col(i) = face_normal;
                }
            }

            virtual ~Cell() = default;

            // setters and getters
            int getDimension() const { return dim; }
            int getIndex() const { return index; }
            // void setIndex(int idx) { index = idx; }
            const Eigen::MatrixXd& getOutwardNormVec() const { return outward_norm_vec; }
            Eigen::VectorXd getOutwardNormVec(int i) const {
                if (i < 0 || i >= outward_norm_vec.cols()) {
                    throw std::out_of_range("Outward normal vector index out of range");
                }
                return outward_norm_vec.col(i);
            }
            double getMeasure() const { return measure; }
            const Node<dim>& getCentroid() const { return centroid; }
            // Jacobian matrix
            const Eigen::MatrixXd& getJacobianMatrix() const { return JacobianMatrix; }
            // vertex
            const std::vector<TypedNodePtr>& getVertices() const { return vertices; }
            const TypedNodePtr& getVertex(int i) const {
                if (i < 0 || i >= vertices.size()) {
                    throw std::out_of_range("Vertex index out of range");
                }
                return vertices[i];
            }
            Eigen::MatrixXd getVerticesCoordinates() const {
                Eigen::MatrixXd vertices_coordinates = Eigen::MatrixXd::Zero(dim, vertices.size());
                for(int i = 0; i < vertices.size(); i++){
                    vertices_coordinates.col(i) = vertices[i]->getCoordinates();
                }
                return vertices_coordinates;
            }

            // face
            const std::vector<TypedFacePtr>& getFaces() const { return faces; }
            const TypedFacePtr& getFace(int i) const {
                if (i < 0 || i >= faces.size()) {
                    throw std::out_of_range("Face index out of range");
                }
                return faces[i];
            }
            void setFaces(const std::vector<TypedFacePtr>& faces) {
                if (faces.size() != this->faces.size()) {
                    throw std::invalid_argument("Number of faces must match");
                }
                this->faces = faces;
            }
    
            void setFace(int i, const TypedFacePtr& face) { 
                if (i < 0 || i >= faces.size()) {
                    throw std::out_of_range("Face index out of range");
                }
                faces[i] = face; 
            }

            // adjacent cell
            const std::vector<int>& getAdjacentCells() const { return adjacent_cells; }
            int getAdjacentCell(int i) const {
                if (i < 0 || i >= adjacent_cells.size()) {
                    throw std::out_of_range("Adjacent cell index out of range");
                }
                return adjacent_cells[i];
            }
            void setAdjacentCells(const std::vector<int>& cells) {
                if (cells.size() != faces.size()) {
                    throw std::invalid_argument("Number of adjacent cells must match number of faces");
                }
                adjacent_cells = cells;
            }
            void setAdjacentCell(int i, int cell_index) {
                if (i < 0 || i >= adjacent_cells.size()) {
                    throw std::out_of_range("Adjacent cell index out of range");
                }
                adjacent_cells[i] = cell_index;
            }
            
            // operators
            bool isPointInside(const TypedNodePtr& point, double tolerance = 1.0e-12) const {
                
                if constexpr (dim == 2) {
                    const auto& n0 = *std::static_pointer_cast<Node<2>>(vertices[0]);
                    const auto& n1 = *std::static_pointer_cast<Node<2>>(vertices[1]);
                    const auto& n2 = *std::static_pointer_cast<Node<2>>(vertices[2]);
                    const auto& p = *std::static_pointer_cast<Node<2>>(point);
    
                    Eigen::Vector2d v0 = n1.getCoordinates() - n0.getCoordinates();
                    Eigen::Vector2d v1 = n2.getCoordinates() - n0.getCoordinates();
                    Eigen::Vector2d v2 = p.getCoordinates() - n0.getCoordinates();
    
                    double d00 = v0.dot(v0);
                    double d01 = v0.dot(v1);
                    double d11 = v1.dot(v1);
                    double d20 = v2.dot(v0);
                    double d21 = v2.dot(v1);
    
                    double denom = d00 * d11 - d01 * d01;
                    
                    // Scale-dependent tolerance: use relative tolerance based on cell size
                    double max_edge_sq = std::max(d00, d11);
                    double denom_tol = tolerance * max_edge_sq * max_edge_sq;
                    if (std::abs(denom) < denom_tol) {
                        return false;
                    }
                    
                    double w = (d11 * d20 - d01 * d21) / denom;
                    double v = (d00 * d21 - d01 * d20) / denom;
                    double u = 1.0 - v - w;
    
                    // Use relative tolerance for barycentric coordinates
                    double coord_tol = tolerance * std::sqrt(max_edge_sq);
                    return u >= -coord_tol && v >= -coord_tol && w >= -coord_tol;
                }
                else if constexpr (dim == 3) {
                    const auto& n0 = *std::static_pointer_cast<Node<3>>(vertices[0]);
                    const auto& n1 = *std::static_pointer_cast<Node<3>>(vertices[1]);
                    const auto& n2 = *std::static_pointer_cast<Node<3>>(vertices[2]);
                    const auto& n3 = *std::static_pointer_cast<Node<3>>(vertices[3]);
                    const auto& p = *std::static_pointer_cast<Node<3>>(point);
    
                    Eigen::Vector3d v0 = n1.getCoordinates() - n0.getCoordinates();
                    Eigen::Vector3d v1 = n2.getCoordinates() - n0.getCoordinates();
                    Eigen::Vector3d v2 = n3.getCoordinates() - n0.getCoordinates();
                    Eigen::Vector3d v3 = p.getCoordinates() - n0.getCoordinates();
    
                    double V = v0.cross(v1).dot(v2);
                    
                    // Scale-dependent tolerance: use relative tolerance based on cell size
                    double max_edge_sq = std::max({v0.squaredNorm(), v1.squaredNorm(), v2.squaredNorm()});
                    double V_tol = tolerance * std::pow(max_edge_sq, 1.5);
                    if (std::abs(V) < V_tol) {
                        return false; 
                    }
    
                    double V0 = v3.cross(v1).dot(v2);
                    double V1 = v0.cross(v3).dot(v2);
                    double V2 = v0.cross(v1).dot(v3);
    
                    double a = V0 / V;
                    double b = V1 / V;
                    double c = V2 / V;
                    double d = 1.0 - a - b - c;
    
                    // Use relative tolerance for barycentric coordinates
                    double coord_tol = tolerance * std::sqrt(max_edge_sq);
                    return a >= -coord_tol && b >= -coord_tol && c >= -coord_tol && d >= -coord_tol;
                }
                else{
                    throw std::runtime_error("Can't check if point is inside cell for dynamic dimension cell.");
                }
            }

            bool operator==(const Cell& other) const {
                return vertices == other.vertices && faces == other.faces && index == other.index;
            }

            bool operator!=(const Cell& other) const {
                return !(*this == other);
            }
    };

}