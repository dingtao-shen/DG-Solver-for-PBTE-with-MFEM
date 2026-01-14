#pragma once

#include "Eigen/Dense"
#include "SpatialMesh/Element.hpp"
#include "SpatialMesh/MeshPartitioning.hpp"
#include "GlobalConfig/GlobalConfig.hpp"
#include "SolidAngle/SolidAngle.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <memory>
#include <iomanip> // Required for std::fixed and std::setprecision

namespace SpatialMesh {

    template <int dim>
    class SpatialMesh {
        static_assert(dim == 2 || dim == 3, "Mesh dimension must be 2 or 3");

        private:
            using TypedNodePtr = std::conditional_t<dim == 2, Node2DPtr, Node3DPtr>;
            using TypedFacePtr = std::conditional_t<dim == 2, Face2DPtr, Face3DPtr>;
            using TypedCellPtr = std::conditional_t<dim == 2, Cell2DPtr, Cell3DPtr>;

            std::vector<TypedNodePtr> nodes;
            std::vector<TypedFacePtr> faces;
            std::vector<TypedCellPtr> cells;
            std::map<int, std::string> boundary_faces;
            std::vector<std::vector<std::vector<int>>> computation_order;
            std::vector<std::vector<Eigen::MatrixXd>> outflow;
            std::unique_ptr<MeshPartitionInfo<dim>> partition_info;

        public:
            virtual ~SpatialMesh() = default;

            SpatialMesh() = default;
            explicit SpatialMesh(const std::string& filename, SolidAngle& dsa);
            
            // 禁用复制构造函数和赋值操作符，因为包含unique_ptr
            SpatialMesh(const SpatialMesh&) = delete;
            SpatialMesh& operator=(const SpatialMesh&) = delete;
            
            // 允许移动构造函数和移动赋值操作符
            SpatialMesh(SpatialMesh&&) = default;
            SpatialMesh& operator=(SpatialMesh&&) = default;

            // Getters
            static constexpr int getDimension() { return dim; }
            size_t getNumVertices() const { return nodes.size(); }
            size_t getNumFaces() const { return faces.size(); }
            size_t getNumCells() const { return cells.size(); }
            const std::vector<TypedNodePtr>& getNodes() const { return nodes; }
            const std::vector<TypedFacePtr>& getFaces() const { return faces; }
            const std::vector<TypedCellPtr>& getCells() const { return cells; }
            const std::map<int, std::string>& getBoundaryFaces() const { return boundary_faces; }
            const std::vector<std::vector<std::vector<int>>>& getComputationOrder() const { return computation_order; }
            const std::vector<std::vector<Eigen::MatrixXd>>& getOutflow() const { return outflow; }
            void setupComputationOrder(const SolidAngle& dsa);
            std::vector<int> CalComputationOrder(Eigen::VectorXd& Vi);
            
            void computeOutflow(const SolidAngle& dsa);

            // Mesh partitioning
            bool partitionMesh(int num_partitions);
            const MeshPartitionInfo<dim>* getPartitionInfo() const { return partition_info.get(); }
            bool isPartitioned() const { return partition_info != nullptr; }

            // Save mesh information to file
            void SaveMeshInfo(std::string filename) const;

        protected:
            // Mesh connectivity
            virtual void connectFacesToCells();

    };

    template <int dim>
    SpatialMesh<dim>::SpatialMesh(const std::string& filename, SolidAngle& dsa) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open mesh file: " + filename);
        }

        std::string line;
        Eigen::MatrixXd vertices_matrix;
        Eigen::MatrixXi elements;
        std::unordered_map<int, int> node_pairs;
        std::vector<std::unordered_map<int, int>> bd_pairs;
        int num_boundary_faces = 0;
        int num_nodes = 0;
        int num_elements = 0;
        int bS, bM, nS, nM, Npairs;

        std::cout << ">>> Loading mesh file: " << filename << std::endl;

        while (std::getline(file, line)) {
            if (line.empty()) continue;
            if (line.find("$MeshFormat") != std::string::npos) {
                std::getline(file, line);
                std::istringstream iss(line);
                double version;
                int n_binary, n_float;
                iss >> version >> n_binary >> n_float;
                if(version != 2.2 || n_binary != 0 || n_float != 8){
                    throw std::runtime_error("Mesh format not supported");
                }
                std::getline(file, line); // $EndMeshFormat
            }
            else if (line.find("$PhysicalNames") != std::string::npos) {
                std::getline(file, line);
                num_boundary_faces = std::stoi(line);
                for (int i = 0; i < num_boundary_faces; ++i) {
                    std::getline(file, line);
                    std::istringstream iss(line);
                    int dimension, tag;
                    std::string name;
                    iss >> dimension >> tag >> name;
                    name = name.substr(1, name.length() - 2);
                    boundary_faces[tag] = name;
                }
                std::getline(file, line); // $EndPhysicalNames
            }
            else if (line.find("$Nodes") != std::string::npos) {
                std::getline(file, line);
                num_nodes = std::stoi(line);
                vertices_matrix.resize(num_nodes, dim);
                
                for (int i = 0; i < num_nodes; ++i) {
                    std::getline(file, line);
                    std::istringstream iss(line);
                    int node_id;
                    Eigen::Vector3d coords;
                    iss >> node_id >> coords(0) >> coords(1) >> coords(2);
                    for(int j = 0; j < dim; j++){
                        vertices_matrix(node_id - 1, j) = coords(j);
                    }
                }

                std::getline(file, line); // $EndNodes
            }
            else if (line.find("$Elements") != std::string::npos) {
                std::getline(file, line); // Read number of elements
                num_elements = std::stoi(line);
                elements.resize(num_elements, dim + 3);

                for (int i = 0; i < num_elements; ++i) {
                    std::getline(file, line);
                    std::istringstream iss(line);
                    int elemId, dummy;
                    iss >> elemId;
                    elements.row(elemId - 1).setConstant(-1);
                    iss >> elements(elemId - 1, 0) >> dummy >> elements(elemId - 1, 1) >> dummy;
                    // Read node indices
                    std::vector<int> nodeIndices;
                    int nodeIdx, j = 2;
                    while (iss >> nodeIdx) {
                        elements(elemId - 1, j) = nodeIdx - 1; // Convert to 0-based indexing
                        j++;
                    }
                }

                std::getline(file, line); // Read $EndElements
            }
            else if (line.find("$Periodic") != std::string::npos) { // not checked yet
                int dummy;
                file >> dummy;
                file >> dummy >> bS >> bM;
                bd_pairs.push_back(std::unordered_map<int, int>());
                bd_pairs.back()[bS] = bM;
                bd_pairs.back()[bM] = bS;
                file >> line;
                for(int d = 0; d < 16; d++){file >> dummy;}
                file >> Npairs;
                for(int i = 0; i < Npairs; i++){
                    file >> nS >> nM;
                    node_pairs[nS - 1] = nM - 1;
                    node_pairs[nM - 1] = nS - 1;
                }
                std::getline(file, line); // Read $EndPeriodic
            }
        }

        nodes.clear();
        faces.clear();
        cells.clear();

        // Scale coordinates from nondimensional to physical units using L_REF
        if (vertices_matrix.size() > 0) {
            vertices_matrix = vertices_matrix * CC.L_REF;
        }

        nodes.reserve(num_nodes);
        for (int i = 0; i < num_nodes; ++i) {
            Eigen::VectorXd coords = vertices_matrix.row(i);
            nodes.push_back(std::make_shared<Node<dim>>(
                coords,
                i
            ));
        }
        
        for (int i = 0; i < num_elements; ++i) {
            const auto& elemRow = elements.row(i);
            int face_tag = dim == 2 ? 1 : 2;
            int cell_tag = dim == 2 ? 2 : 4;
            if (elemRow(0) == face_tag) {
                int boundary_flag = 0;
                if (boundary_faces.find(elemRow(1)) != boundary_faces.end()) {
                    boundary_flag = elemRow(1);
                }
                std::vector<TypedNodePtr> face_vertices;
                face_vertices.reserve(dim);
                for(int j = 2; j < 2 + dim; j++){
                    face_vertices.push_back(nodes[elemRow(j)]);
                }
                faces.push_back(std::make_shared<Face<dim, dim>>(
                    face_vertices,
                    static_cast<int>(faces.size()),
                    boundary_flag
                ));
            }
            else if (elemRow(0) == cell_tag) {
                std::vector<TypedNodePtr> cell_vertices;
                cell_vertices.reserve(dim+1);
                for(int j = 2; j <= 2 + dim; j++){
                    cell_vertices.push_back(nodes[elemRow(j)]);
                }
                cells.push_back(std::make_shared<Cell<dim, dim + 1>>(
                    cell_vertices,
                    static_cast<int>(cells.size())
                ));
                std::vector<std::vector<int>> fv_idx;
                fv_idx.reserve(dim+1);
                for(int j = 0; j < dim + 1; j++){
                    std::vector<int> fv_idx_;
                    for(int k = 0; k < dim; k++){
                        fv_idx_.push_back((j+k) % (dim+1) + 2);
                    }
                    fv_idx.push_back(fv_idx_);
                }
                for (const auto& fv : fv_idx) {
                    bool face_exists = false;
                    for (const auto& face : faces) {
                        bool found = true;
                        for(int k = 0; k < dim; k++){
                            if(!face->hasVertex(nodes[elemRow(fv[k])])){
                                found = false;
                                break;
                            }
                        }
                        if (found) {
                            face_exists = true;
                            break;
                        }
                    }
                    if (!face_exists) {
                        std::vector<TypedNodePtr> face_vertices;
                        face_vertices.reserve(dim);
                        for(int k = 0; k < dim; k++){
                            face_vertices.push_back(nodes[elemRow(fv[k])]);
                        }
                        faces.push_back(std::make_shared<Face<dim, dim>>(
                            face_vertices,
                            static_cast<int>(faces.size())
                        ));
                    }
                }
            }
        }

        connectFacesToCells();
        setupComputationOrder(dsa);
        computeOutflow(dsa);

        // match periodic faces
        if (!node_pairs.empty() && !bd_pairs.empty()) {
            // Create a map to store face pairs based on boundary pairs
            std::unordered_map<int, int> face_pairs;
            
            // For each boundary pair group
            // Find faces that belong to these boundary tags
            std::vector<int> faces_in_group;
            for (size_t i = 0; i < faces.size(); ++i) {
                if (CC.BOUNDARY_COND[faces[i]->getBoundaryTag()].first == 4) {
                    faces_in_group.push_back(i);
                }
            }
            
            // Match faces based on node pairs
            for (size_t i = 0; i < faces_in_group.size(); ++i) {
                int face1_idx = faces_in_group[i];
                const auto& vertices1 = faces[face1_idx]->getVertices();
                std::set<int> nodes_set1;
                for (const auto& node : vertices1) {
                    int node1_idx = node->getIndex();
                    auto it = node_pairs.find(node1_idx);
                    if (it != node_pairs.end()) {
                        nodes_set1.insert(it->second);
                    }
                    else{
                        throw std::runtime_error("Node " + std::to_string(node1_idx) + " not found in node_pairs");
                    }
                }

                for (size_t j = 0; j < faces_in_group.size(); ++j) {
                    int face2_idx = faces_in_group[j];
                    if (face1_idx == face2_idx) continue;

                    const auto& vertices2 = faces[face2_idx]->getVertices();
                    if (vertices1.size() != vertices2.size()) continue;

                    std::set<int> nodes_set2;
                    for (const auto& node : vertices2) {
                        nodes_set2.insert(node->getIndex());
                    }

                    if (nodes_set1 == nodes_set2){
                        face_pairs[face1_idx] = face2_idx;
                        face_pairs[face2_idx] = face1_idx;
                        faces[face1_idx]->setPeriodicPair(face2_idx);
                        faces[face2_idx]->setPeriodicPair(face1_idx);
                        int k;
                        k = (faces[face1_idx]->getAdjacentCells()[0] == -1) ? 0 : 1;
                        faces[face1_idx]->setAdjacentCell(k, faces[face2_idx]->getIndex());
                        k = (faces[face2_idx]->getAdjacentCells()[0] == -1) ? 0 : 1;
                        faces[face2_idx]->setAdjacentCell(k, faces[face1_idx]->getIndex());
                        break;
                    }
                    
                }
            }
        }

        std::cout << "  >>>" << dim << "D Spatial mesh created successfully" << std::endl;
        std::cout << "  - Nodes: " << nodes.size() << std::endl;
        std::cout << "  - Faces: " << faces.size() << std::endl;
        std::cout << "  - Cells: " << cells.size() << std::endl;
    }

    template <int dim>
    void SpatialMesh<dim>::connectFacesToCells() {
        int nf = dim == 2 ? 3 : 4;

        for (size_t i = 0; i < cells.size(); i++) {
            auto cell = cells[i];
            const auto& vertices = cell->getVertices();
            std::vector<int> fv_idx;
            for (size_t j = 0; j < nf; j++) {
                for(size_t k = 0; k < nf - 1; k++){
                    fv_idx.push_back((j+k) % nf);
                }
                for (size_t k = 0; k < faces.size(); k++) {
                    auto face = faces[k];
                    bool found = true;
                    for(size_t l = 0; l < nf - 1; l++){
                        if(!face->hasVertex(vertices[fv_idx[l]])){
                            found = false;
                            break;
                        }
                    }
                    if (found) {
                        cell->setFace(j, face);
                        auto adj_cells = face->getAdjacentCells();
                        size_t pos = adj_cells[0] == -1 ? 0 : 1;
                        face->setAdjacentCell(pos, static_cast<int>(i));
                        face->setLocalIndex(pos, static_cast<int>(j));
                        break;
                    }
                }
                fv_idx.clear();
            }
        }

        for (size_t i = 0; i < cells.size(); i++) {
            auto cell = cells[i];
            std::vector<int> adjacent_cells(nf, -1);
            
            for (size_t j = 0; j < nf; j++) {
                const auto& face = cell->getFace(j);
                const auto& face_adj_cells = face->getAdjacentCells();
                
                for (int adj_cell_idx : face_adj_cells) {
                    if (adj_cell_idx != -1 && adj_cell_idx != static_cast<int>(i)) {
                        adjacent_cells[j] = adj_cell_idx;
                        break;
                    }
                }
            }
            cell->setAdjacentCells(adjacent_cells);
        }

        for (const auto& cell : cells){
            int index = cell->getIndex();
            std::vector<int> adjacent_cells(nf, -1);
            for(size_t i = 0; i < nf; ++i) {
                const auto& face = cell->getFace(i);
                const auto& face_adj_cells = face->getAdjacentCells();
                for (int adj_cell_idx : face_adj_cells) {
                    if (adj_cell_idx != -1 && adj_cell_idx != index) {
                        adjacent_cells[i] = adj_cell_idx;
                        break;
                    }
                }
            }
            cell->setAdjacentCells(adjacent_cells);
        }
    }

    template <int dim>
    void SpatialMesh<dim>::setupComputationOrder(const SolidAngle& dsa) {
        computation_order.resize(CC.NPOLE, std::vector<std::vector<int>>(CC.NAZIM, std::vector<int>(cells.size(), -1)));

        // Pre-compute directions
        std::vector<std::vector<Eigen::VectorXd>> directions(CC.NPOLE, std::vector<Eigen::VectorXd>(CC.NAZIM, Eigen::VectorXd::Zero(dim)));
        for (int j1 = 0; j1 < CC.NPOLE; ++j1) {
            for (int j2 = 0; j2 < CC.NAZIM; ++j2) {
                for(int i = 0; i < dim; i++){
                    directions[j1][j2](i) = dsa.dir()[j1][j2](i);
                }
            }
        }

        // Compute order for each angle
        for (int j1 = 0; j1 < CC.NPOLE; ++j1) {
            for (int j2 = 0; j2 < CC.NAZIM; ++j2) {
                std::vector<bool> processed(cells.size(), false);
                size_t count = 0;

                while (count < cells.size()) {
                    for (size_t i = 0; i < cells.size(); ++i) {
                        if (processed[i]) continue;

                        bool ready = true;
                        const auto& cell = cells[i];
                        
                        // Check all faces
                        for (size_t f = 0; f < dim + 1; ++f) {
                            const auto& face = cell->getFace(f);
                            const auto& adj_cells = face->getAdjacentCells();
                            
                            // Find neighbor cell
                            int neighbor_idx = -1;
                            for (int adj : adj_cells) {
                                if (adj != -1 && adj != static_cast<int>(i)) {
                                    neighbor_idx = adj;
                                    break;
                                }
                            }

                            if (neighbor_idx != -1 && !processed[neighbor_idx]) {
                                const auto& norm = cell->getOutwardNormVec(f);
                                if (norm.dot(directions[j1][j2]) < 0) {
                                    ready = false;
                                    break;
                                }
                            }
                        }

                        if (ready) {
                            computation_order[j1][j2][count++] = i;
                            processed[i] = true;
                        }
                    }
                }
            }
        }
    }
    
    template <int dim>
    std::vector<int> SpatialMesh<dim>::CalComputationOrder(Eigen::VectorXd& Vi) {
        std::vector<int> comp_order_vi(cells.size(), -1);

        // Compute order for each angle

        std::vector<bool> processed(cells.size(), false);
        size_t count = 0;

        while (count < cells.size()) {
            for (size_t i = 0; i < cells.size(); ++i) {
                if (processed[i]) continue;

                bool ready = true;
                const auto& cell = cells[i];
                
                // Check all faces
                for (size_t f = 0; f < dim + 1; ++f) {
                    const auto& face = cell->getFace(f);
                    const auto& adj_cells = face->getAdjacentCells();
                    
                    // Find neighbor cell
                    int neighbor_idx = -1;
                    for (int adj : adj_cells) {
                        if (adj != -1 && adj != static_cast<int>(i)) {
                            neighbor_idx = adj;
                            break;
                        }
                    }

                    if (neighbor_idx != -1 && !processed[neighbor_idx]) {
                        const auto& norm = cell->getOutwardNormVec(f);
                        if (norm.dot(Vi) < 0) {
                            ready = false;
                            break;
                        }
                    }
                }

                if (ready) {
                    comp_order_vi[count++] = i;
                    processed[i] = true;
                }
            }
        }
        return comp_order_vi;
    }

    template <int dim>
    void SpatialMesh<dim>::computeOutflow(const SolidAngle& dsa) {
        outflow.resize(CC.NPOLE, std::vector<Eigen::MatrixXd>(CC.NAZIM, Eigen::MatrixXd::Zero(dim + 1, cells.size())));
        const auto dir_all = dsa.dir();
        // #pragma omp parallel for collapse(3)
        for(int j1 = 0; j1 < CC.NPOLE; j1++){
            for(int j2 = 0; j2 < CC.NAZIM; j2++){
                for(int l = 0; l < cells.size(); l++){
                    const auto norm = cells[l]->getOutwardNormVec();
                    for(int t = 0; t < CC.SPATIAL_DIM+1; t++){
                        double s = 0.0;
                        for(int i = 0; i < CC.SPATIAL_DIM; i++){
                            s += dir_all[j1][j2](i) * norm(i, t);
                        }
                        outflow[j1][j2](t, l) = s;
                    }
                }
            }
        }
    }

    template <int dim>
    void SpatialMesh<dim>::SaveMeshInfo(std::string filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open mesh file: " + filename);
        }

        // Write header information
        file << "# ========================================" << std::endl;
        file << "# " << dim << "D Spatial Mesh Information" << std::endl;
        file << "# Generated by DG-Solver-for-PhononBTE" << std::endl;
        file << "# ========================================" << std::endl;
        file << std::endl;

        // Write mesh statistics
        file << "# Mesh Statistics:" << std::endl;
        file << "# - Total Nodes: " << nodes.size() << std::endl;
        file << "# - Total Faces: " << faces.size() << std::endl;
        file << "# - Total Cells: " << cells.size() << std::endl;
        file << "# - Dimension: " << dim << "D" << std::endl;
        file << std::endl;

        // Write boundary face information
        file << "# Boundary Face Information:" << std::endl;
        for (const auto& [tag, name] : boundary_faces) {
            file << "# - Boundary " << tag << ": " << name << std::endl;
        }
        file << std::endl;

        // Write detailed cell information in the format matching Test.cpp output
        file << "# ========================================" << std::endl;
        file << "# DETAILED CELL INFORMATION" << std::endl;
        file << "# ========================================" << std::endl;
        file << std::endl;

        for (const auto& cell : cells) {
            int CellID = cell->getIndex();
            file << "CellID = " << CellID << std::endl;
            file << "Centroid = " << cell->getCentroid().getCoordinates().transpose() << std::endl;
            file << "Measure = " << cell->getMeasure() << std::endl;
            file << "********** ********** ********** ********** ********** **********" << std::endl;
            
            // Write vertex information
            for (size_t i = 0; i < cell->getVertices().size(); ++i) {
                const auto& vertex = cell->getVertices()[i];
                file << "Vertices " << vertex->getIndex() << ": " << vertex->getCoordinates().transpose() << std::endl;
            }
            
            file << "********** ********** ********** ********** ********** **********" << std::endl;
            
            // Write outward normal vectors for faces
            const auto& outward_norms = cell->getOutwardNormVec();
            for (size_t i = 0; i < outward_norms.cols(); ++i) {
                const auto& face = cell->getFaces()[i];
                file << "Outward Normal for face: " << face->getIndex() << " = " << outward_norms.col(i).transpose() << std::endl;
            }
            file << "********** ********** ********** ********** ********** **********" << std::endl;
            file << std::endl;
        }

        file << "# ========================================" << std::endl;
        file << "# END OF MESH INFORMATION" << std::endl;
        file << "# ========================================" << std::endl;

        file.close();
        std::cout << "  >>>Mesh information saved to: " << filename << std::endl;
    }

    /******************************************************************************
    ***************************** Mesh partitioning ******************************
    ******************************************************************************/

    template <int dim>
    bool SpatialMesh<dim>::partitionMesh(int num_partitions) {
        if (num_partitions <= 0) {
            std::cerr << "Error: Number of partitions must be positive" << std::endl;
            return false;
        }

        if (cells.empty()) {
            std::cerr << "Error: No cells in mesh for partitioning" << std::endl;
            return false;
        }

        partition_info = std::make_unique<MeshPartitionInfo<dim>>(num_partitions);

        bool success = MeshPartitioner<dim>::partitionMesh(*this, num_partitions, *partition_info);

        if (success) {
            std::cout << "  >>>Mesh partitioned successfully into " << num_partitions << " parts" << std::endl;
            partition_info->printPartitionStatistics();
        } else {
            std::cerr << "  >>>Mesh partitioning failed" << std::endl;
            partition_info.reset();
        }

        return success;
    }

    // MeshPartitioner implementation
    template <int dim>
    bool MeshPartitioner<dim>::partitionMesh(const SpatialMesh<dim>& mesh, 
                                            int num_partitions, 
                                            MeshPartitionInfo<dim>& partition_info) {
        
        if (num_partitions <= 0) {
            std::cerr << "Error: Number of partitions must be positive" << std::endl;
            return false;
        }

        if (num_partitions == 1) {
            // 单分区情况
            partition_info = MeshPartitionInfo<dim>(1);
            std::vector<idx_t> single_partition(mesh.getNumCells(), 0);
            partition_info.setCellPartition(single_partition);
            for (int i = 0; i < mesh.getNumCells(); ++i) {
                partition_info.addCellToPartition(i, 0);
            }
            return true;
        }

        std::vector<idx_t> xadj, adjncy, vwgt, adjwgt;
        if (!buildMetisGraph(mesh, xadj, adjncy, vwgt, adjwgt)) {
            std::cerr << "Error: Failed to build METIS graph" << std::endl;
            return false;
        }

        std::vector<idx_t> cell_partition(mesh.getNumCells());
        idx_t nvtxs = static_cast<idx_t>(mesh.getNumCells());
        idx_t ncon = 1;  // number of constraints
        idx_t nparts = static_cast<idx_t>(num_partitions);
        idx_t objval;
        std::vector<idx_t> part(nvtxs);

        // set METIS options
        idx_t options[METIS_NOPTIONS];
        METIS_SetDefaultOptions(options);
        options[METIS_OPTION_PTYPE] = METIS_PTYPE_KWAY;  // use k-way partitioning
        options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;  // minimize edge cuts
        options[METIS_OPTION_CTYPE] = METIS_CTYPE_SHEM;  // use SHEM connectivity
        options[METIS_OPTION_IPTYPE] = METIS_IPTYPE_GROW;  // use GROW initial partitioning
        options[METIS_OPTION_RTYPE] = METIS_RTYPE_FM;  // use FM optimization
        options[METIS_OPTION_NCUTS] = 1;  // number of runs
        options[METIS_OPTION_NITER] = 10;  // number of iterations
        options[METIS_OPTION_UFACTOR] = 30;  // load balancing factor

        // execute partitioning
        int ret = METIS_PartGraphKway(&nvtxs, &ncon, xadj.data(), adjncy.data(),
                                    vwgt.empty() ? nullptr : vwgt.data(),
                                    nullptr,  // vsize
                                    adjwgt.empty() ? nullptr : adjwgt.data(),
                                    &nparts, nullptr,  // tpwgts
                                    nullptr,  // ubvec
                                    options, &objval, part.data());

        if (ret != METIS_OK) {
            std::cerr << "Error: METIS partitioning failed with code " << ret << std::endl;
            return false;
        }

        // create partition info object
        partition_info = MeshPartitionInfo<dim>(num_partitions);
        partition_info.setCellPartition(std::vector<idx_t>(part.begin(), part.end()));

        // fill partition info
        fillPartitionInfo(mesh, partition_info);

        std::cout << "Mesh partitioned successfully into " << num_partitions 
                 << " parts with " << objval << " edge cuts" << std::endl;

        return true;
    }

    template <int dim>
    bool MeshPartitioner<dim>::buildMetisGraph(const SpatialMesh<dim>& mesh,
                                              std::vector<idx_t>& xadj,
                                              std::vector<idx_t>& adjncy,
                                              std::vector<idx_t>& vwgt,
                                              std::vector<idx_t>& adjwgt) {
        
        const auto& cells = mesh.getCells();
        size_t num_cells = cells.size();

        if (num_cells == 0) {
            std::cerr << "Error: No cells in mesh" << std::endl;
            return false;
        }

        // build cell neighbors
        std::vector<std::vector<int>> cell_neighbors(num_cells);
        
        for (size_t i = 0; i < num_cells; ++i) {
            const auto& cell = cells[i];
            const auto& adjacent_cells = cell->getAdjacentCells();
            
            for (int adj_cell : adjacent_cells) {
                if (adj_cell != -1 && static_cast<size_t>(adj_cell) < num_cells) {
                    bool found = false;
                    for (int neighbor : cell_neighbors[i]) {
                        if (neighbor == adj_cell) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        cell_neighbors[i].push_back(adj_cell);
                    }
                }
            }
        }

        // build METIS format adjacency table
        xadj.clear();
        adjncy.clear();
        xadj.reserve(num_cells + 1);
        adjncy.reserve(num_cells * 6);  // estimated size

        xadj.push_back(0);
        for (size_t i = 0; i < num_cells; ++i) {
            for (int neighbor : cell_neighbors[i]) {
                adjncy.push_back(static_cast<idx_t>(neighbor));
            }
            xadj.push_back(static_cast<idx_t>(adjncy.size()));
        }

        // set vertex weights (optional, here using cell area/volume as weights)
        vwgt.clear();
        vwgt.reserve(num_cells);
        for (const auto& cell : cells) {
            // // convert area/volume to integer weights
            // double weight = cell->getMeasure() * 1000.0;  // multiply by 1000
            vwgt.push_back(static_cast<idx_t>(1.0));
        }

        // set edge weights (optional, here using 1)
        adjwgt.clear();
        adjwgt.resize(adjncy.size(), 1);

        return true;
    }

    template <int dim>
    void MeshPartitioner<dim>::fillPartitionInfo(const SpatialMesh<dim>& mesh, 
                                                MeshPartitionInfo<dim>& partition_info) {
        
        const auto& cells = mesh.getCells();
        const auto& faces = mesh.getFaces();
        const auto& cell_partition = partition_info.getCellPartition();

        // assign cells to corresponding partitions
        for (size_t i = 0; i < cells.size(); ++i) {
            int partition_id = static_cast<int>(cell_partition[i]);
            partition_info.addCellToPartition(static_cast<int>(i), partition_id);
        }

        // compute computation order for each partition
        const auto& mesh_computation_order = mesh.getComputationOrder();
        int num_partitions = partition_info.getNumPartitions();
        
        // Initialize partition_computation_order_ for each partition
        for (int p = 0; p < num_partitions; ++p) {
            partition_info.initializePartitionComputationOrder(p, CC.NPOLE, CC.NAZIM);
        }
        
        // Generate computation order for each partition
        for (int p = 0; p < num_partitions; ++p) {
            for (int j1 = 0; j1 < CC.NPOLE; ++j1) {
                for (int j2 = 0; j2 < CC.NAZIM; ++j2) {
                    std::vector<int> partition_order;
                    // Iterate through mesh computation order and extract cells belonging to this partition
                    for (size_t idx = 0; idx < mesh_computation_order[j1][j2].size(); ++idx) {
                        int cell_id = mesh_computation_order[j1][j2][idx];
                        if (cell_id >= 0 && cell_id < static_cast<int>(cell_partition.size())) {
                            if (static_cast<int>(cell_partition[cell_id]) == p) {
                                partition_order.push_back(cell_id);
                            }
                        }
                    }
                    partition_info.setPartitionComputationOrder(p, j1, j2, partition_order);
                }
            }
        }

        // identify boundary faces and communication information
        for (size_t i = 0; i < faces.size(); ++i) {
            const auto& face = faces[i];
            const auto& adjacent_cells = face->getAdjacentCells();
            
            if (face->isBoundary()) {
                // boundary faces: add to corresponding partitions
                for (int cell_id : adjacent_cells) {
                    if (cell_id != -1) {
                        int partition_id = static_cast<int>(cell_partition[cell_id]);
                        partition_info.addBoundaryFace(partition_id, static_cast<int>(i));
                    }
                }
            } else {
                // internal faces: check if they cross partitions
                std::vector<int> face_partitions;
                for (int cell_id : adjacent_cells) {
                    if (cell_id != -1) {
                        int partition_id = static_cast<int>(cell_partition[cell_id]);
                        if (std::find(face_partitions.begin(), face_partitions.end(), 
                                    partition_id) == face_partitions.end()) {
                            face_partitions.push_back(partition_id);
                        }
                    }
                }
                
                // if face crosses multiple partitions, add to communication information
                if (face_partitions.size() > 1) {
                    for (size_t j = 0; j < face_partitions.size(); ++j) {
                        for (size_t k = 0; k < face_partitions.size(); ++k) {
                            if (j != k) {
                                partition_info.addCommunicationFace(
                                    face_partitions[j], face_partitions[k], static_cast<int>(i));
                            }
                        }
                    }
                }
            }
        }

        // identify nbr cells and communication cells
        for (int i = 0; i < cells.size(); ++i) {
            const auto& cell = cells[i];
            const auto& adjacent_cells = cell->getAdjacentCells();
            for (int adj_cell : adjacent_cells) {
                if (adj_cell != -1 && cell_partition[adj_cell] != cell_partition[i]) {
                    partition_info.addNbrCell(cell_partition[i], adj_cell);
                    partition_info.addCommunicationCell(cell_partition[i], i, adj_cell);
                }
            }
        }

        for(int i = 0; i < partition_info.getNumPartitions(); i++){
            for(int j = 0; j < partition_info.getPartitionCells(i).size(); j++){
                partition_info.setLocalCellIdx(i, partition_info.getPartitionCells(i)[j], j);
            }
            for(int j = 0; j < partition_info.getPartitionNbrCells(i).size(); j++){
                partition_info.setLocalNbrCellIdx(i, partition_info.getPartitionNbrCells(i)[j], j);
            }
            // finalize precomputed communication maps for each partition
            partition_info.finalizeCommunicationMaps(i);
        }


    }

} // namespace SpatialMesh