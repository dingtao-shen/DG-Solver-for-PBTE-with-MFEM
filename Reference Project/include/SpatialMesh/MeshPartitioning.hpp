#pragma once

#include "metis.h"
#include <vector>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <stdexcept>
#include <iostream>

// Forward declarations
namespace SpatialMesh {
    template<int dim> class SpatialMesh;
}

namespace SpatialMesh {

    template <int dim>
    class MeshPartitionInfo {
    private:
        int num_partitions_;
        std::vector<idx_t> cell_partition_;  // each cell's partition id
        std::vector<std::vector<int>> partition_cells_;  // each partition's cells
        std::vector<std::vector<int>> boundary_faces_;  // each partition's boundary faces
        std::vector<std::map<int, std::vector<int>>> partition_communication_;  // each partition's communication faces
        std::vector<std::vector<int>> partition_nbr_cells_; // each partition's nbr cells
        std::vector<std::map<int, std::vector<int>>> partition_communication_cells_;  // each partition's communication cells
        std::vector<std::map<int, int>> local_cell_idx_;
        std::vector<std::map<int, int>> local_nbr_cell_idx_;
        // Precomputed for communication efficiency
        // For each partition: dest_rank -> unique local cell IDs to send (sorted)
        std::vector<std::unordered_map<int, std::vector<int>>> partition_dest_cells_;
        // For each partition: source_rank -> remote cell IDs to receive (sorted)
        std::vector<std::unordered_map<int, std::vector<int>>> partition_recv_cells_;
        // For each partition: source_rank -> local neighbor buffer indices aligned with partition_recv_cells_
        std::vector<std::unordered_map<int, std::vector<int>>> partition_recv_nbr_indices_;
        // For each partition: list of unique source ranks expected to receive from (sorted)
        std::vector<std::vector<int>> partition_source_ranks_;

        std::vector<std::vector<std::vector<std::vector<int>>>> partition_computation_order_;
        
    public:
        MeshPartitionInfo() = default;
        explicit MeshPartitionInfo(int num_partitions) : num_partitions_(num_partitions) {
            partition_cells_.resize(num_partitions_);
            boundary_faces_.resize(num_partitions_);
            partition_communication_.resize(num_partitions_);
            partition_communication_cells_.resize(num_partitions_);
            partition_nbr_cells_.resize(num_partitions_);
            partition_communication_cells_.resize(num_partitions_);
            local_cell_idx_.resize(num_partitions_);
            local_nbr_cell_idx_.resize(num_partitions_);
            partition_dest_cells_.resize(num_partitions_);
            partition_recv_cells_.resize(num_partitions_);
            partition_recv_nbr_indices_.resize(num_partitions_);
            partition_source_ranks_.resize(num_partitions_);
            partition_computation_order_.resize(num_partitions_);
        }

        // Getters
        int getNumPartitions() const { return num_partitions_; }
        const std::vector<idx_t>& getCellPartition() const { return cell_partition_; }
        idx_t getCellPartition(int cell_id) const { 
            if (cell_id < 0 || cell_id >= static_cast<int>(cell_partition_.size())) {
                throw std::out_of_range("Cell ID out of range");
            }
            return cell_partition_[cell_id]; 
        }
        const std::vector<std::vector<int>>& getPartitionCells() const { return partition_cells_; }
        const std::vector<int>& getPartitionCells(int partition_id) const {
            if (partition_id < 0 || partition_id >= num_partitions_) {
                throw std::out_of_range("Partition ID out of range");
            }
            return partition_cells_[partition_id];
        }
        const std::vector<std::vector<int>>& getBoundaryFaces() const { return boundary_faces_; }
        const std::vector<int>& getBoundaryFaces(int partition_id) const {
            if (partition_id < 0 || partition_id >= num_partitions_) {
                throw std::out_of_range("Partition ID out of range");
            }
            return boundary_faces_[partition_id];
        }
        const std::vector<std::map<int, std::vector<int>>>& getPartitionCommunication() const { 
            return partition_communication_; 
        }
        const std::map<int, std::vector<int>>& getPartitionCommunication(int partition_id) const {
            if (partition_id < 0 || partition_id >= num_partitions_) {
                throw std::out_of_range("Partition ID out of range");
            }
            return partition_communication_[partition_id];
        }
        const std::vector<std::vector<int>>& getPartitionNbrCells() const { return partition_nbr_cells_; }
        const std::vector<int>& getPartitionNbrCells(int partition_id) const {
            if (partition_id < 0 || partition_id >= num_partitions_) {
                throw std::out_of_range("Partition ID out of range");
            }
            return partition_nbr_cells_[partition_id];
        }
        const std::vector<std::map<int, std::vector<int>>>& getPartitionCommunicationCells() const { return partition_communication_cells_; }
        const std::map<int, std::vector<int>>& getPartitionCommunicationCells(int partition_id) const {
            if (partition_id < 0 || partition_id >= num_partitions_) {
                throw std::out_of_range("Partition ID out of range");
            }
            return partition_communication_cells_[partition_id];
        }
        const std::map<int, int>& getLocalCellIdx(int partition_id) const {
            if (partition_id < 0 || partition_id >= num_partitions_) {
                throw std::out_of_range("Partition ID out of range");
            }
            return local_cell_idx_[partition_id];
        }
        const std::map<int, int>& getLocalNbrCellIdx(int partition_id) const {
            if (partition_id < 0 || partition_id >= num_partitions_) {
                throw std::out_of_range("Partition ID out of range");
            }
            return local_nbr_cell_idx_[partition_id];
        }
        const std::unordered_map<int, std::vector<int>>& getPartitionDestCells(int partition_id) const {
            if (partition_id < 0 || partition_id >= num_partitions_) {
                throw std::out_of_range("Partition ID out of range");
            }
            return partition_dest_cells_[partition_id];
        }
        const std::unordered_map<int, std::vector<int>>& getPartitionRecvCells(int partition_id) const {
            if (partition_id < 0 || partition_id >= num_partitions_) {
                throw std::out_of_range("Partition ID out of range");
            }
            return partition_recv_cells_[partition_id];
        }
        const std::unordered_map<int, std::vector<int>>& getPartitionRecvNbrIndices(int partition_id) const {
            if (partition_id < 0 || partition_id >= num_partitions_) {
                throw std::out_of_range("Partition ID out of range");
            }
            return partition_recv_nbr_indices_[partition_id];
        }
        const std::vector<int>& getPartitionSourceRanks(int partition_id) const {
            if (partition_id < 0 || partition_id >= num_partitions_) {
                throw std::out_of_range("Partition ID out of range");
            }
            return partition_source_ranks_[partition_id];
        }
        const std::vector<std::vector<std::vector<std::vector<int>>>>& getPartitionComputationOrder() const {
            return partition_computation_order_;
        }
        const std::vector<std::vector<std::vector<int>>>& getPartitionComputationOrder(int partition_id) const {
            if (partition_id < 0 || partition_id >= num_partitions_) {
                throw std::out_of_range("Partition ID out of range");
            }
            return partition_computation_order_[partition_id];
        }

        // Setters
        void setCellPartition(const std::vector<idx_t>& partition) { cell_partition_ = partition; }
        void setCellPartition(int cell_id, idx_t partition_id) {
            if (cell_id < 0 || cell_id >= static_cast<int>(cell_partition_.size())) {
                throw std::out_of_range("Cell ID out of range");
            }
            cell_partition_[cell_id] = partition_id;
        }
        void addCellToPartition(int cell_id, int partition_id) {
            if (partition_id < 0 || partition_id >= num_partitions_) {
                throw std::out_of_range("Partition ID out of range");
            }
            partition_cells_[partition_id].push_back(cell_id);
        }
        void addBoundaryFace(int partition_id, int face_id) {
            if (partition_id < 0 || partition_id >= num_partitions_) {
                throw std::out_of_range("Partition ID out of range");
            }
            boundary_faces_[partition_id].push_back(face_id);
        }
        void addCommunicationFace(int from_partition, int to_partition, int face_id) {
            if (from_partition < 0 || from_partition >= num_partitions_ ||
                to_partition < 0 || to_partition >= num_partitions_) {
                throw std::out_of_range("Partition ID out of range");
            }
            partition_communication_[from_partition][to_partition].push_back(face_id);
        }
        void addNbrCell(int partition_id, int nbr_cell_id) {
            if (partition_id < 0 || partition_id >= num_partitions_) {
                throw std::out_of_range("Partition ID out of range");
            }
            if (std::find(partition_nbr_cells_[partition_id].begin(), partition_nbr_cells_[partition_id].end(), nbr_cell_id) == partition_nbr_cells_[partition_id].end()) {
                partition_nbr_cells_[partition_id].push_back(nbr_cell_id);
            }
        }
        void addCommunicationCell(int partition_id, int local_cell_id, int nbr_cell_id) {
            if (partition_id < 0 || partition_id >= num_partitions_) {
                throw std::out_of_range("Partition ID out of range");
            }
            if (partition_communication_cells_[partition_id].find(local_cell_id) == partition_communication_cells_[partition_id].end()) {
                partition_communication_cells_[partition_id].insert({local_cell_id, {nbr_cell_id}});
            }
            else if (std::find(partition_communication_cells_[partition_id][local_cell_id].begin(), partition_communication_cells_[partition_id][local_cell_id].end(), nbr_cell_id) == partition_communication_cells_[partition_id][local_cell_id].end()) {
                partition_communication_cells_[partition_id][local_cell_id].push_back(nbr_cell_id);
            }
        }
        void setLocalCellIdx(int partition_id, int local_cell_id, int idx) {
            if (partition_id < 0 || partition_id >= num_partitions_) {
                throw std::out_of_range("Partition ID out of range");
            }
            if (cell_partition_[local_cell_id] != partition_id) {
                throw std::invalid_argument("Local cell ID does not belong to the partition");
            }
            local_cell_idx_[partition_id][local_cell_id] = idx;
        }
        void setLocalNbrCellIdx(int partition_id, int nbr_cell_id, int idx) {
            if (partition_id < 0 || partition_id >= num_partitions_) {
                throw std::out_of_range("Partition ID out of range");
            }
            local_nbr_cell_idx_[partition_id][nbr_cell_id] = idx;
        }
        void setPartitionComputationOrder(int partition_id, int j1, int j2, const std::vector<int>& order) {
            if (partition_id < 0 || partition_id >= num_partitions_) {
                throw std::out_of_range("Partition ID out of range");
            }
            if (j1 < 0 || j1 >= static_cast<int>(partition_computation_order_[partition_id].size())) {
                partition_computation_order_[partition_id].resize(j1 + 1);
            }
            if (j2 < 0 || j2 >= static_cast<int>(partition_computation_order_[partition_id][j1].size())) {
                partition_computation_order_[partition_id][j1].resize(j2 + 1);
            }
            partition_computation_order_[partition_id][j1][j2] = order;
        }
        void initializePartitionComputationOrder(int partition_id, int npole, int nazim) {
            if (partition_id < 0 || partition_id >= num_partitions_) {
                throw std::out_of_range("Partition ID out of range");
            }
            partition_computation_order_[partition_id].resize(npole);
            for (int j1 = 0; j1 < npole; ++j1) {
                partition_computation_order_[partition_id][j1].resize(nazim);
            }
        }
        // Build precomputed dest cells and source ranks for a partition
        void finalizeCommunicationMaps(int partition_id) {
            if (partition_id < 0 || partition_id >= num_partitions_) {
                throw std::out_of_range("Partition ID out of range");
            }
            // dest cells
            std::unordered_map<int, std::vector<int>> dest_to_cells;
            std::unordered_map<int, std::unordered_set<int>> dest_unique;
            for (const auto& kv : partition_communication_cells_[partition_id]) {
                const int local_cell_id = kv.first;
                for (const int nbr_cell : kv.second) {
                    int dest_rank = static_cast<int>(cell_partition_[nbr_cell]);
                    if (dest_rank == partition_id) continue;
                    auto& inserted = dest_unique[dest_rank];
                    if (inserted.insert(local_cell_id).second) {
                        dest_to_cells[dest_rank].push_back(local_cell_id);
                    }
                }
            }
            for (auto& kv : dest_to_cells) {
                auto& vec = kv.second;
                std::sort(vec.begin(), vec.end());
            }
            partition_dest_cells_[partition_id] = std::move(dest_to_cells);

            // source ranks, receive cells, and target indices
            std::unordered_map<int, std::vector<int>> src_to_cells;
            std::unordered_map<int, std::vector<int>> src_to_indices;
            std::unordered_set<int> source_set;
            for (const int nbr_cell_id : partition_nbr_cells_[partition_id]) {
                int src = static_cast<int>(cell_partition_[nbr_cell_id]);
                if (src == partition_id) continue;
                source_set.insert(src);
                src_to_cells[src].push_back(nbr_cell_id);
                auto it_idx = local_nbr_cell_idx_[partition_id].find(nbr_cell_id);
                if (it_idx == local_nbr_cell_idx_[partition_id].end()) {
                    throw std::runtime_error("Neighbor cell index not found while finalizing communication maps");
                }
                src_to_indices[src].push_back(it_idx->second);
            }
            for (auto& kv : src_to_cells) {
                const int src_rank = kv.first;
                auto& cells = kv.second;
                auto& indices = src_to_indices[src_rank];
                std::vector<std::pair<int, int>> paired;
                paired.reserve(cells.size());
                for (size_t i = 0; i < cells.size(); ++i) {
                    paired.emplace_back(cells[i], indices[i]);
                }
                std::sort(paired.begin(), paired.end(),
                          [](const auto& a, const auto& b) { return a.first < b.first; });
                for (size_t i = 0; i < paired.size(); ++i) {
                    cells[i] = paired[i].first;
                    indices[i] = paired[i].second;
                }
            }
            partition_recv_cells_[partition_id] = std::move(src_to_cells);
            partition_recv_nbr_indices_[partition_id] = std::move(src_to_indices);

            std::vector<int> sources(source_set.begin(), source_set.end());
            std::sort(sources.begin(), sources.end());
            partition_source_ranks_[partition_id] = std::move(sources);
        }
        // print partition statistics
        void printPartitionStatistics() const {
            std::cout << "=== Mesh Partition Statistics ===" << std::endl;
            std::cout << "Number of partitions: " << num_partitions_ << std::endl;
            
            for (int i = 0; i < num_partitions_; ++i) {
                std::cout << "Partition " << i << ":" << std::endl;
                std::cout << "  - Cells: " << partition_cells_[i].size() << std::endl;
                std::cout << "  - Boundary faces: " << boundary_faces_[i].size() << std::endl;
                std::cout << "  - Communication with " << partition_communication_[i].size() 
                         << " other partitions" << std::endl;
            }
            std::cout << "================================" << std::endl;
        }
    };

    template <int dim>
    class MeshPartitioner {
    public:
        static bool partitionMesh(const SpatialMesh<dim>& mesh, 
                                 int num_partitions, 
                                 MeshPartitionInfo<dim>& partition_info);
    private:
        static bool buildMetisGraph(const SpatialMesh<dim>& mesh,
                                  std::vector<idx_t>& xadj,
                                  std::vector<idx_t>& adjncy,
                                  std::vector<idx_t>& vwgt,
                                  std::vector<idx_t>& adjwgt);
        static void fillPartitionInfo(const SpatialMesh<dim>& mesh, 
                                    MeshPartitionInfo<dim>& partition_info);
    };

} // namespace SpatialMesh
