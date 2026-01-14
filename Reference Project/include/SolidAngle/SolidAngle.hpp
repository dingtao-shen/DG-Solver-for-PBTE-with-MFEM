#pragma once

#include "GlobalConfig/GlobalConfig.hpp"
#include "Eigen/Dense"

class SolidAngle
{
private:
    std::vector<std::vector<Eigen::VectorXd>> direction;
    std::vector<std::vector<double>> weight;
public:
    SolidAngle(int dim, int npole, int nazim, int pattern);
    ~SolidAngle();
    std::vector<std::vector<Eigen::VectorXd>> dir() const;
    std::vector<std::vector<double>> wt() const;
    void output();
};