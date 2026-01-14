#pragma once

#include "SpatialMesh/SpatialMesh.hpp"
#include "SpatialMesh/MeshPartitioning.hpp"
#include <vector>
#include <set>
#include <map>
#include <string>
#include <iostream>

namespace Validation {

    template <int dim>
    class MeshPartitionValidator {
    public:
        struct ValidationResult {
            bool is_valid;
            std::vector<std::string> errors;
            std::vector<std::string> warnings;
            
            ValidationResult() : is_valid(true) {}
            
            void addError(const std::string& error) {
                is_valid = false;
                errors.push_back(error);
            }
            
            void addWarning(const std::string& warning) {
                warnings.push_back(warning);
            }
            
            void print() const {
                std::cout << "=== Mesh Partition Validation Results ===" << std::endl;
                if (is_valid) {
                    std::cout << "✓ Validation PASSED" << std::endl;
                } else {
                    std::cout << "✗ Validation FAILED" << std::endl;
                }
                
                if (!errors.empty()) {
                    std::cout << "\nErrors (" << errors.size() << "):" << std::endl;
                    for (const auto& error : errors) {
                        std::cout << "  ✗ " << error << std::endl;
                    }
                }
                
                if (!warnings.empty()) {
                    std::cout << "\nWarnings (" << warnings.size() << "):" << std::endl;
                    for (const auto& warning : warnings) {
                        std::cout << "  ⚠ " << warning << std::endl;
                    }
                }
                std::cout << "==========================================" << std::endl;
            }
        };
        
        // Main validation function
        static ValidationResult validate(const SpatialMesh::SpatialMesh<dim>& mesh,
                                        const SpatialMesh::MeshPartitionInfo<dim>& partition_info);
        
    private:
        // Individual validation checks
        static bool validateCellPartitionAssignment(
            const SpatialMesh::SpatialMesh<dim>& mesh,
            const SpatialMesh::MeshPartitionInfo<dim>& partition_info,
            ValidationResult& result);
        
        static bool validatePartitionCellsConsistency(
            const SpatialMesh::SpatialMesh<dim>& mesh,
            const SpatialMesh::MeshPartitionInfo<dim>& partition_info,
            ValidationResult& result);
        
        static bool validateBoundaryFaces(
            const SpatialMesh::SpatialMesh<dim>& mesh,
            const SpatialMesh::MeshPartitionInfo<dim>& partition_info,
            ValidationResult& result);
        
        static bool validateCommunicationFaces(
            const SpatialMesh::SpatialMesh<dim>& mesh,
            const SpatialMesh::MeshPartitionInfo<dim>& partition_info,
            ValidationResult& result);
        
        static bool validateNeighborCells(
            const SpatialMesh::SpatialMesh<dim>& mesh,
            const SpatialMesh::MeshPartitionInfo<dim>& partition_info,
            ValidationResult& result);
        
        static bool validateCommunicationCells(
            const SpatialMesh::SpatialMesh<dim>& mesh,
            const SpatialMesh::MeshPartitionInfo<dim>& partition_info,
            ValidationResult& result);
        
        static bool validateLocalIndices(
            const SpatialMesh::SpatialMesh<dim>& mesh,
            const SpatialMesh::MeshPartitionInfo<dim>& partition_info,
            ValidationResult& result);
    };

} // namespace Validation

