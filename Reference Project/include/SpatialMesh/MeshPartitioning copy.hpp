#pragma once

#include "metis.h"
#include <vector>
#include <map>
#include <unordered_map>
#include <memory>
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
        std::vector<idx_t> cell_partition_;  // 每个单元的分区ID
        std::vector<std::vector<int>> partition_cells_;  // 每个分区包含的单元
        std::vector<std::vector<int>> nbr_cells_; // 每个分区需要通信的邻居单元id列表
        std::vector<std::map<int, std::vector<int>>> partition_communication_cells_;  // 分区间通信的单元对， key: 需要通信的本分区单元id，value: 位于其他分区的邻居单元id的列表
        std::vector<std::vector<int>> boundary_faces_;  // 每个分区的边界面
        std::vector<std::map<int, std::vector<int>>> partition_communication_;  // 分区间通信信息
        
    public:
        MeshPartitionInfo() = default;
        explicit MeshPartitionInfo(int num_partitions) : num_partitions_(num_partitions) {
            partition_cells_.resize(num_partitions_);
            boundary_faces_.resize(num_partitions_);
            partition_communication_.resize(num_partitions_);
            partition_communication_cells_.resize(num_partitions_);
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

        // 统计信息
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

    /**
     * @brief 网格分区器类
     * 
     * 使用METIS库对网格进行分区
     * 实现将在SpatialMesh.hpp中提供
     */
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
