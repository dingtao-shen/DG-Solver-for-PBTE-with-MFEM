#include "Validation/MeshPartitionValidator.hpp"
#include <algorithm>
#include <sstream>

namespace Validation {

    template <int dim>
    typename MeshPartitionValidator<dim>::ValidationResult
    MeshPartitionValidator<dim>::validate(
        const SpatialMesh::SpatialMesh<dim>& mesh,
        const SpatialMesh::MeshPartitionInfo<dim>& partition_info) {
        
        ValidationResult result;
        
        std::cout << "\n>>> Starting mesh partition validation..." << std::endl;
        
        // Run all validation checks
        validateCellPartitionAssignment(mesh, partition_info, result);
        validatePartitionCellsConsistency(mesh, partition_info, result);
        validateBoundaryFaces(mesh, partition_info, result);
        validateCommunicationFaces(mesh, partition_info, result);
        validateNeighborCells(mesh, partition_info, result);
        validateCommunicationCells(mesh, partition_info, result);
        validateLocalIndices(mesh, partition_info, result);
        
        return result;
    }

    template <int dim>
    bool MeshPartitionValidator<dim>::validateCellPartitionAssignment(
        const SpatialMesh::SpatialMesh<dim>& mesh,
        const SpatialMesh::MeshPartitionInfo<dim>& partition_info,
        ValidationResult& result) {
        
        const auto& cell_partition = partition_info.getCellPartition();
        int num_cells = static_cast<int>(mesh.getNumCells());
        int num_partitions = partition_info.getNumPartitions();
        
        // Check 1: All cells should have a partition assignment
        if (cell_partition.size() != static_cast<size_t>(num_cells)) {
            std::ostringstream oss;
            oss << "Cell partition vector size mismatch: expected " << num_cells 
                << ", got " << cell_partition.size();
            result.addError(oss.str());
            return false;
        }
        
        // Check 2: All partition IDs should be valid
        std::vector<int> partition_cell_counts(num_partitions, 0);
        for (int i = 0; i < num_cells; ++i) {
            int partition_id = static_cast<int>(cell_partition[i]);
            if (partition_id < 0 || partition_id >= num_partitions) {
                std::ostringstream oss;
                oss << "Cell " << i << " has invalid partition ID: " << partition_id;
                result.addError(oss.str());
                return false;
            }
            partition_cell_counts[partition_id]++;
        }
        
        // Check 3: All partitions should have at least one cell
        for (int i = 0; i < num_partitions; ++i) {
            if (partition_cell_counts[i] == 0) {
                std::ostringstream oss;
                oss << "Partition " << i << " has no cells assigned";
                result.addWarning(oss.str());
            }
        }
        
        std::cout << "  ✓ Cell partition assignment validation passed" << std::endl;
        return true;
    }

    template <int dim>
    bool MeshPartitionValidator<dim>::validatePartitionCellsConsistency(
        const SpatialMesh::SpatialMesh<dim>& mesh,
        const SpatialMesh::MeshPartitionInfo<dim>& partition_info,
        ValidationResult& result) {
        
        const auto& cell_partition = partition_info.getCellPartition();
        int num_partitions = partition_info.getNumPartitions();
        
        // Check: partition_cells_ should be consistent with cell_partition_
        std::vector<std::set<int>> partition_cells_sets(num_partitions);
        
        // Build sets from partition_cells_
        for (int i = 0; i < num_partitions; ++i) {
            const auto& cells = partition_info.getPartitionCells(i);
            for (int cell_id : cells) {
                partition_cells_sets[i].insert(cell_id);
            }
        }
        
        // Verify consistency
        for (int cell_id = 0; cell_id < static_cast<int>(cell_partition.size()); ++cell_id) {
            int partition_id = static_cast<int>(cell_partition[cell_id]);
            
            // Check if cell is in the correct partition's cell list
            if (partition_cells_sets[partition_id].find(cell_id) == partition_cells_sets[partition_id].end()) {
                std::ostringstream oss;
                oss << "Cell " << cell_id << " is assigned to partition " << partition_id
                    << " but not found in partition's cell list";
                result.addError(oss.str());
                return false;
            }
            
            // Check if cell is not in other partitions
            for (int i = 0; i < num_partitions; ++i) {
                if (i != partition_id && partition_cells_sets[i].find(cell_id) != partition_cells_sets[i].end()) {
                    std::ostringstream oss;
                    oss << "Cell " << cell_id << " appears in multiple partitions: " 
                        << partition_id << " and " << i;
                    result.addError(oss.str());
                    return false;
            }
            }
        }
        
        // Check: All cells in partition_cells_ should have correct partition assignment
        for (int i = 0; i < num_partitions; ++i) {
            const auto& cells = partition_info.getPartitionCells(i);
            for (int cell_id : cells) {
                if (cell_id < 0 || cell_id >= static_cast<int>(cell_partition.size())) {
                    std::ostringstream oss;
                    oss << "Partition " << i << " contains invalid cell ID: " << cell_id;
                    result.addError(oss.str());
                    return false;
                }
                if (static_cast<int>(cell_partition[cell_id]) != i) {
                    std::ostringstream oss;
                    oss << "Cell " << cell_id << " in partition " << i 
                        << " has partition assignment " << cell_partition[cell_id];
                    result.addError(oss.str());
                    return false;
                }
            }
        }
        
        std::cout << "  ✓ Partition cells consistency validation passed" << std::endl;
        return true;
    }

    template <int dim>
    bool MeshPartitionValidator<dim>::validateBoundaryFaces(
        const SpatialMesh::SpatialMesh<dim>& mesh,
        const SpatialMesh::MeshPartitionInfo<dim>& partition_info,
        ValidationResult& result) {
        
        const auto& faces = mesh.getFaces();
        const auto& cell_partition = partition_info.getCellPartition();
        int num_partitions = partition_info.getNumPartitions();
        
        // Build expected boundary faces for each partition
        std::vector<std::set<int>> expected_boundary_faces(num_partitions);
        
        for (size_t face_id = 0; face_id < faces.size(); ++face_id) {
            const auto& face = faces[face_id];
            if (face->isBoundary()) {
                const auto& adjacent_cells = face->getAdjacentCells();
                for (int cell_id : adjacent_cells) {
                    if (cell_id != -1) {
                        int partition_id = static_cast<int>(cell_partition[cell_id]);
                        expected_boundary_faces[partition_id].insert(static_cast<int>(face_id));
                    }
                }
            }
        }
        
        // Verify boundary faces in partition_info
        for (int i = 0; i < num_partitions; ++i) {
            const auto& boundary_faces = partition_info.getBoundaryFaces(i);
            std::set<int> reported_faces(boundary_faces.begin(), boundary_faces.end());
            
            // Check if all reported boundary faces are actually boundary faces
            for (int face_id : boundary_faces) {
                if (face_id < 0 || face_id >= static_cast<int>(faces.size())) {
                    std::ostringstream oss;
                    oss << "Partition " << i << " has invalid boundary face ID: " << face_id;
                    result.addError(oss.str());
                    return false;
                }
                if (!faces[face_id]->isBoundary()) {
                    std::ostringstream oss;
                    oss << "Partition " << i << " reports face " << face_id 
                        << " as boundary, but it is not a boundary face";
                    result.addError(oss.str());
                    return false;
                }
            }
            
            // Check if all expected boundary faces are reported
            for (int face_id : expected_boundary_faces[i]) {
                if (reported_faces.find(face_id) == reported_faces.end()) {
                    std::ostringstream oss;
                    oss << "Partition " << i << " missing boundary face " << face_id;
                    result.addWarning(oss.str());
                }
            }
        }
        
        std::cout << "  ✓ Boundary faces validation passed" << std::endl;
        return true;
    }

    template <int dim>
    bool MeshPartitionValidator<dim>::validateCommunicationFaces(
        const SpatialMesh::SpatialMesh<dim>& mesh,
        const SpatialMesh::MeshPartitionInfo<dim>& partition_info,
        ValidationResult& result) {
        
        const auto& faces = mesh.getFaces();
        const auto& cell_partition = partition_info.getCellPartition();
        int num_partitions = partition_info.getNumPartitions();
        
        // Build expected communication faces
        std::vector<std::map<int, std::set<int>>> expected_comm_faces(num_partitions);
        
        for (size_t face_id = 0; face_id < faces.size(); ++face_id) {
            const auto& face = faces[face_id];
            if (!face->isBoundary()) {
                const auto& adjacent_cells = face->getAdjacentCells();
                std::vector<int> face_partitions;
                
                for (int cell_id : adjacent_cells) {
                    if (cell_id != -1) {
                        int partition_id = static_cast<int>(cell_partition[cell_id]);
                        if (std::find(face_partitions.begin(), face_partitions.end(), partition_id) 
                            == face_partitions.end()) {
                            face_partitions.push_back(partition_id);
                        }
                    }
                }
                
                // If face spans multiple partitions, it's a communication face
                if (face_partitions.size() > 1) {
                    for (size_t i = 0; i < face_partitions.size(); ++i) {
                        for (size_t j = 0; j < face_partitions.size(); ++j) {
                            if (i != j) {
                                expected_comm_faces[face_partitions[i]][face_partitions[j]]
                                    .insert(static_cast<int>(face_id));
                            }
                        }
                    }
                }
            }
        }
        
        // Verify communication faces in partition_info
        for (int i = 0; i < num_partitions; ++i) {
            const auto& comm_faces = partition_info.getPartitionCommunication(i);
            
            for (const auto& [dest_partition, face_list] : comm_faces) {
                if (dest_partition < 0 || dest_partition >= num_partitions) {
                    std::ostringstream oss;
                    oss << "Partition " << i << " has invalid destination partition: " << dest_partition;
                    result.addError(oss.str());
                    return false;
                }
                
                for (int face_id : face_list) {
                    if (face_id < 0 || face_id >= static_cast<int>(faces.size())) {
                        std::ostringstream oss;
                        oss << "Partition " << i << " has invalid communication face ID: " << face_id;
                        result.addError(oss.str());
                        return false;
                    }
                    
                    const auto& face = faces[face_id];
                    if (face->isBoundary()) {
                        std::ostringstream oss;
                        oss << "Partition " << i << " reports boundary face " << face_id 
                            << " as communication face";
                        result.addError(oss.str());
                        return false;
                    }
                    
                    // Check if face actually connects these partitions
                    const auto& adjacent_cells = face->getAdjacentCells();
                    bool connects_partitions = false;
                    for (int cell_id : adjacent_cells) {
                        if (cell_id != -1) {
                            int cell_partition_id = static_cast<int>(cell_partition[cell_id]);
                            if (cell_partition_id == i || cell_partition_id == dest_partition) {
                                connects_partitions = true;
                                break;
                            }
                        }
                    }
                    
                    if (!connects_partitions) {
                        std::ostringstream oss;
                        oss << "Face " << face_id << " does not connect partitions " 
                            << i << " and " << dest_partition;
                        result.addWarning(oss.str());
                    }
                }
            }
        }
        
        std::cout << "  ✓ Communication faces validation passed" << std::endl;
        return true;
    }

    template <int dim>
    bool MeshPartitionValidator<dim>::validateNeighborCells(
        const SpatialMesh::SpatialMesh<dim>& mesh,
        const SpatialMesh::MeshPartitionInfo<dim>& partition_info,
        ValidationResult& result) {
        
        const auto& cells = mesh.getCells();
        const auto& cell_partition = partition_info.getCellPartition();
        int num_partitions = partition_info.getNumPartitions();
        
        // Build expected neighbor cells for each partition
        std::vector<std::set<int>> expected_nbr_cells(num_partitions);
        
        for (int cell_id = 0; cell_id < static_cast<int>(cells.size()); ++cell_id) {
            const auto& cell = cells[cell_id];
            int cell_partition_id = static_cast<int>(cell_partition[cell_id]);
            const auto& adjacent_cells = cell->getAdjacentCells();
            
            for (int adj_cell_id : adjacent_cells) {
                if (adj_cell_id != -1) {
                    int adj_partition_id = static_cast<int>(cell_partition[adj_cell_id]);
                    if (adj_partition_id != cell_partition_id) {
                        expected_nbr_cells[cell_partition_id].insert(adj_cell_id);
                    }
                }
            }
        }
        
        // Verify neighbor cells in partition_info
        for (int i = 0; i < num_partitions; ++i) {
            const auto& nbr_cells = partition_info.getPartitionNbrCells(i);
            std::set<int> reported_nbr_cells(nbr_cells.begin(), nbr_cells.end());
            
            // Check if all reported neighbor cells are actually neighbors
            for (int nbr_cell_id : nbr_cells) {
                if (nbr_cell_id < 0 || nbr_cell_id >= static_cast<int>(cells.size())) {
                    std::ostringstream oss;
                    oss << "Partition " << i << " has invalid neighbor cell ID: " << nbr_cell_id;
                    result.addError(oss.str());
                    return false;
                }
                
                int nbr_partition_id = static_cast<int>(cell_partition[nbr_cell_id]);
                if (nbr_partition_id == i) {
                    std::ostringstream oss;
                    oss << "Partition " << i << " reports its own cell " << nbr_cell_id 
                        << " as neighbor cell";
                    result.addError(oss.str());
                    return false;
                }
            }
            
            // Check if all expected neighbor cells are reported
            for (int nbr_cell_id : expected_nbr_cells[i]) {
                if (reported_nbr_cells.find(nbr_cell_id) == reported_nbr_cells.end()) {
                    std::ostringstream oss;
                    oss << "Partition " << i << " missing neighbor cell " << nbr_cell_id;
                    result.addWarning(oss.str());
                }
            }
        }
        
        std::cout << "  ✓ Neighbor cells validation passed" << std::endl;
        return true;
    }

    template <int dim>
    bool MeshPartitionValidator<dim>::validateCommunicationCells(
        const SpatialMesh::SpatialMesh<dim>& mesh,
        const SpatialMesh::MeshPartitionInfo<dim>& partition_info,
        ValidationResult& result) {
        
        const auto& cells = mesh.getCells();
        const auto& cell_partition = partition_info.getCellPartition();
        int num_partitions = partition_info.getNumPartitions();
        
        // Build expected communication cells
        std::vector<std::map<int, std::set<int>>> expected_comm_cells(num_partitions);
        
        for (int cell_id = 0; cell_id < static_cast<int>(cells.size()); ++cell_id) {
            const auto& cell = cells[cell_id];
            int cell_partition_id = static_cast<int>(cell_partition[cell_id]);
            const auto& adjacent_cells = cell->getAdjacentCells();
            
            for (int adj_cell_id : adjacent_cells) {
                if (adj_cell_id != -1) {
                    int adj_partition_id = static_cast<int>(cell_partition[adj_cell_id]);
                    if (adj_partition_id != cell_partition_id) {
                        expected_comm_cells[cell_partition_id][cell_id].insert(adj_cell_id);
                    }
                }
            }
        }
        
        // Verify communication cells in partition_info
        for (int i = 0; i < num_partitions; ++i) {
            const auto& comm_cells = partition_info.getPartitionCommunicationCells(i);
            
            for (const auto& [local_cell_id, nbr_cell_list] : comm_cells) {
                // Check local cell
                if (local_cell_id < 0 || local_cell_id >= static_cast<int>(cells.size())) {
                    std::ostringstream oss;
                    oss << "Partition " << i << " has invalid local cell ID: " << local_cell_id;
                    result.addError(oss.str());
                    return false;
                }
                
                int local_partition_id = static_cast<int>(cell_partition[local_cell_id]);
                if (local_partition_id != i) {
                    std::ostringstream oss;
                    oss << "Partition " << i << " reports cell " << local_cell_id 
                        << " as local cell, but it belongs to partition " << local_partition_id;
                    result.addError(oss.str());
                    return false;
                }
                
                // Check neighbor cells
                for (int nbr_cell_id : nbr_cell_list) {
                    if (nbr_cell_id < 0 || nbr_cell_id >= static_cast<int>(cells.size())) {
                        std::ostringstream oss;
                        oss << "Partition " << i << " has invalid neighbor cell ID: " << nbr_cell_id;
                        result.addError(oss.str());
                        return false;
                    }
                    
                    int nbr_partition_id = static_cast<int>(cell_partition[nbr_cell_id]);
                    if (nbr_partition_id == i) {
                        std::ostringstream oss;
                        oss << "Partition " << i << " reports its own cell " << nbr_cell_id 
                            << " as neighbor in communication cells";
                        result.addError(oss.str());
                        return false;
                    }
                    
                    // Check if cells are actually neighbors
                    const auto& local_cell = cells[local_cell_id];
                    const auto& adj_cells = local_cell->getAdjacentCells();
                    bool is_neighbor = false;
                    for (int adj_cell : adj_cells) {
                        if (adj_cell == nbr_cell_id) {
                            is_neighbor = true;
                            break;
                        }
                    }
                    
                    if (!is_neighbor) {
                        std::ostringstream oss;
                        oss << "Cell " << local_cell_id << " and cell " << nbr_cell_id 
                            << " are not neighbors";
                        result.addError(oss.str());
                        return false;
                    }
                }
            }
        }
        
        std::cout << "  ✓ Communication cells validation passed" << std::endl;
        return true;
    }

    template <int dim>
    bool MeshPartitionValidator<dim>::validateLocalIndices(
        const SpatialMesh::SpatialMesh<dim>& mesh,
        const SpatialMesh::MeshPartitionInfo<dim>& partition_info,
        ValidationResult& result) {
        
        int num_partitions = partition_info.getNumPartitions();
        
        // Verify local cell indices
        for (int i = 0; i < num_partitions; ++i) {
            const auto& local_cell_idx = partition_info.getLocalCellIdx(i);
            const auto& partition_cells = partition_info.getPartitionCells(i);
            
            // Check if all partition cells have local indices
            for (size_t j = 0; j < partition_cells.size(); ++j) {
                int cell_id = partition_cells[j];
                auto it = local_cell_idx.find(cell_id);
                if (it == local_cell_idx.end()) {
                    std::ostringstream oss;
                    oss << "Partition " << i << " cell " << cell_id 
                        << " missing local index";
                    result.addError(oss.str());
                    return false;
                }
                if (it->second != static_cast<int>(j)) {
                    std::ostringstream oss;
                    oss << "Partition " << i << " cell " << cell_id 
                        << " has incorrect local index: expected " << j 
                        << ", got " << it->second;
                    result.addWarning(oss.str());
                }
            }
            
            // Check if all local indices correspond to partition cells
            for (const auto& [cell_id, local_idx] : local_cell_idx) {
                if (std::find(partition_cells.begin(), partition_cells.end(), cell_id) 
                    == partition_cells.end()) {
                    std::ostringstream oss;
                    oss << "Partition " << i << " has local index for cell " << cell_id 
                        << " which is not in partition";
                    result.addError(oss.str());
                    return false;
                }
            }
        }
        
        // Verify local neighbor cell indices
        for (int i = 0; i < num_partitions; ++i) {
            const auto& local_nbr_cell_idx = partition_info.getLocalNbrCellIdx(i);
            const auto& nbr_cells = partition_info.getPartitionNbrCells(i);
            
            // Check if all neighbor cells have local indices
            for (size_t j = 0; j < nbr_cells.size(); ++j) {
                int nbr_cell_id = nbr_cells[j];
                auto it = local_nbr_cell_idx.find(nbr_cell_id);
                if (it == local_nbr_cell_idx.end()) {
                    std::ostringstream oss;
                    oss << "Partition " << i << " neighbor cell " << nbr_cell_id 
                        << " missing local index";
                    result.addWarning(oss.str());
                } else if (it->second != static_cast<int>(j)) {
                    std::ostringstream oss;
                    oss << "Partition " << i << " neighbor cell " << nbr_cell_id 
                        << " has incorrect local index: expected " << j 
                        << ", got " << it->second;
                    result.addWarning(oss.str());
                }
            }
        }
        
        std::cout << "  ✓ Local indices validation passed" << std::endl;
        return true;
    }

    // Explicit template instantiations
    template class MeshPartitionValidator<2>;
    template class MeshPartitionValidator<3>;

} // namespace Validation

