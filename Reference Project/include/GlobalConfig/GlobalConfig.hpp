#pragma once

#include <yaml-cpp/yaml.h>
#include <unordered_map>
#include <vector>
#include <string>
#include "Eigen/Dense"
#include <iostream>

struct ControlConstants {
    // physics
    double H = 1.054571800e-34;
    double KB = 1.38064852e-23;

    // material
    std::string MATERIAL;
    int MATERIAL_DIM;
    int POLAR_DIM;

    double T_REF;
    double L_REF;

    std::string MODEL;
    int SPATIAL_DIM;
    
    int POLYDEG;
    int DOF;
    int SOLID_ANGLE_PATTERN;
    int NPOLE;
    int NAZIM;
    double OMEGA;
    int NSPEC;

    int TMAX;
    double TOL;
    std::string output_path;

    int MESH_TYPE;
    std::string MESH_FILE;
    int N_BOUNDARY;
    std::unordered_map<int, std::pair<int, double>> BOUNDARY_COND;
    int N_MESH_CELL;

    void LoadFromFile(const std::string& filename){
        YAML::Node config = YAML::LoadFile(filename);

        MATERIAL = config["MATERIAL"].as<std::string>();
        MATERIAL_DIM = config["MATERIAL_DIM"].as<int>();
        POLAR_DIM = config["POLAR_DIM"].as<int>();

        if(MATERIAL_DIM == 2) {
            OMEGA = 2 * M_PI;
        }
        else if(MATERIAL_DIM == 3) {
            OMEGA = 4 * M_PI;
        }
        
        T_REF = config["T_REF"].as<double>();
        L_REF = config["L_REF"].as<double>();

        // model
        MODEL = config["MODEL"].as<std::string>();
        SPATIAL_DIM = config["SPATIAL_DIM"].as<int>();
        
        POLYDEG = config["POLYDEG"].as<int>();
        if(SPATIAL_DIM == 1) {
            DOF = POLYDEG + 1;
        }
        else if(SPATIAL_DIM == 2) {
            DOF = (POLYDEG + 1) * (POLYDEG + 2) / 2;
        }
        else if(SPATIAL_DIM == 3) {
            DOF = (POLYDEG + 1) * (POLYDEG + 2) * (POLYDEG + 3) / 6;
        }

        NPOLE = config["NPOLE"].as<int>();
        NAZIM = config["NAZIM"].as<int>();
        if(MATERIAL_DIM == 2) {
            NPOLE = 1;
        }
        SOLID_ANGLE_PATTERN = config["SOLID_ANGLE_PATTERN"].as<int>();
        NSPEC = config["NSPEC"].as<int>();

        TMAX = config["TMAX"].as<int>();
        TOL = config["TOL"].as<double>();

        // mesh
        MESH_TYPE = config["MESH_TYPE"].as<int>();
        MESH_FILE = config["MESH_PATH"].as<std::string>() + config["MESH_TAG"].as<std::string>() + ".msh";
        N_BOUNDARY = config["N_BOUNDARY"].as<int>();
        for (const auto& boundary : config["BOUNDARY_COND"]) {
            BOUNDARY_COND[boundary.first.as<int>()] = std::make_pair(boundary.second[0].as<int>(), boundary.second[1].as<double>());
        }

        output_path = config["OUTPUT_PATH"].as<std::string>() + MATERIAL + "_" \
         + std::to_string(SPATIAL_DIM) + "D_DEG" + std::to_string(POLYDEG) \
         + "_NP" + std::to_string(NPOLE) + "_NA" + std::to_string(NAZIM) \
         + "MESH" + config["MESH_TAG"].as<std::string>() \
         + ".txt";
    }
};

struct PhononConstants {
    // phonon
    std::vector<double> C_LA;
    std::vector<double> C_TA;
    double LATTICE_DIST;
    std::vector<double> K_RANGE;
    double Ai;
    double BL;
    double BT;
    double BU;

    void LoadFromFile(const std::string& filename){
        YAML::Node config = YAML::LoadFile(filename);

        C_LA = config["C_LA"].as<std::vector<double>>();
        C_TA = config["C_TA"].as<std::vector<double>>();
        LATTICE_DIST = config["LATTICE_DIST"].as<double>();
        K_RANGE.resize(2);
        K_RANGE[0] = 0.0;
        K_RANGE[1] = 2.0 * M_PI / LATTICE_DIST;

        Ai = config["Ai"].as<double>();
        BL = config["BL"].as<double>();
        BT = config["BT"].as<double>();
        BU = config["BU"].as<double>();
    }
};

extern ControlConstants CC;
extern PhononConstants PC;