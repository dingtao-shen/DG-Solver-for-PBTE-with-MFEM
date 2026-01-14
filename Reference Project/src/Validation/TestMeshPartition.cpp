#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include "GlobalConfig/GlobalConfig.hpp"
#include "SpatialMesh/SpatialMesh.hpp"
#include "SolidAngle/SolidAngle.hpp"
#include "Validation/MeshPartitionValidator.hpp"

using namespace Validation;

ControlConstants CC;
PhononConstants PC;

// Helper function to find config file
std::string findConfigFile(const std::string& filename) {
    // Extract just the filename from the path
    std::string basename = filename;
    size_t last_slash = filename.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        basename = filename.substr(last_slash + 1);
    }
    
    // Try multiple possible paths
    std::vector<std::string> possible_paths = {
        filename,  // Original path (from project root)
        "../" + filename,  // One level up (from build/)
        "../config/control/" + basename,  // From build/ to config/control/
        "config/control/" + basename,  // From project root
        "../../config/control/" + basename  // Two levels up
    };
    
    for (const auto& path : possible_paths) {
        std::ifstream file(path);
        if (file.good()) {
            file.close();
            return path;
        }
    }
    
    // If not found, return original path (will throw error later)
    return filename;
}

// Helper function to find mesh file
std::string findMeshFile(const std::string& filename) {
    // Try multiple possible paths
    std::vector<std::string> possible_paths = {
        filename,  // Original path (from project root)
        "../" + filename,  // One level up (from build/)
        "../../" + filename,  // Two levels up
    };
    
    for (const auto& path : possible_paths) {
        std::ifstream file(path);
        if (file.good()) {
            file.close();
            return path;
        }
    }
    
    // If not found, return original path (will throw error later)
    return filename;
}

int main(int argc, char* argv[]) {
    std::cout.precision(12);
    
    // Print usage information
    if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        std::cout << "Usage: " << argv[0] << " [num_partitions]" << std::endl;
        std::cout << "  num_partitions: Number of mesh partitions (default: 4)" << std::endl;
        std::cout << "  -h, --help:    Show this help message" << std::endl;
        std::cout << std::endl;
        std::cout << "Note: Mesh file path is read from Control.yaml configuration file." << std::endl;
        return 0;
    }
    
    // Load configuration
    try {
        std::string phonon_config = findConfigFile("config/control/Si_PhononModel.yaml");
        std::string control_config = findConfigFile("config/control/Control.yaml");
        PC.LoadFromFile(phonon_config);
        CC.LoadFromFile(control_config);
        std::cout << ">>> Parameter loading completed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error loading configuration: " << e.what() << std::endl;
        std::cerr << "Please ensure you run from project root or build directory" << std::endl;
        return 1;
    }
    
    // Get mesh file path from GlobalConfig
    std::string mesh_file = CC.MESH_FILE;
    // Try to find mesh file in multiple possible locations
    mesh_file = findMeshFile(mesh_file);
    
    // Get number of partitions from command line (optional)
    int num_partitions = 4;
    if (argc > 1) {
        num_partitions = std::stoi(argv[1]);
    }
    
    if (num_partitions <= 0) {
        std::cerr << "Error: Number of partitions must be positive" << std::endl;
        return 1;
    }
    
    // Create solid angle
    SolidAngle DSA(CC.MATERIAL_DIM, CC.NPOLE, CC.NAZIM, CC.SOLID_ANGLE_PATTERN);
    
    // Load and construct mesh
    std::cout << "\n>>> Loading and constructing mesh from: " << mesh_file << std::endl;
    SpatialMesh::SpatialMesh<DIM> mesh;
    try {
        mesh = std::move(SpatialMesh::SpatialMesh<DIM>(mesh_file, DSA));
        std::cout << ">>> Mesh loaded successfully!" << std::endl;
        std::cout << "  - Nodes: " << mesh.getNumVertices() << std::endl;
        std::cout << "  - Faces: " << mesh.getNumFaces() << std::endl;
        std::cout << "  - Cells: " << mesh.getNumCells() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error loading mesh: " << e.what() << std::endl;
        return 1;
    }
    
    // Perform METIS partitioning
    std::cout << "\n>>> Starting METIS partitioning into " << num_partitions << " partitions..." << std::endl;
    #ifdef HAVE_METIS
    bool partition_success = mesh.partitionMesh(num_partitions);
    
    if (!partition_success) {
        std::cerr << "Error: Mesh partitioning failed" << std::endl;
        return 1;
    }
    
    const auto* partition_info = mesh.getPartitionInfo();
    if (!partition_info) {
        std::cerr << "Error: Partition info is null" << std::endl;
        return 1;
    }
    
    std::cout << ">>> Mesh partitioned successfully!" << std::endl;
    partition_info->printPartitionStatistics();
    
    // Validate partition information
    std::cout << "\n>>> Starting partition validation..." << std::endl;
    auto validation_result = MeshPartitionValidator<DIM>::validate(mesh, *partition_info);
    
    // Print validation results
    std::cout << std::endl;
    validation_result.print();
    
    // Return exit code based on validation result
    if (validation_result.is_valid) {
        std::cout << "\n>>> All validations passed!" << std::endl;
        return 0;
    } else {
        std::cerr << "\n>>> Validation failed with " << validation_result.errors.size() 
                  << " error(s)!" << std::endl;
        return 1;
    }
    #else
    std::cerr << "Error: METIS is not available. Please install METIS library." << std::endl;
    return 1;
    #endif
}

