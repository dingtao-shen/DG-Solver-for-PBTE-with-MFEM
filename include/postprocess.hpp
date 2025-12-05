#pragma once

#include <string>
#include <mfem.hpp>

namespace dg {

inline void SaveVTK(const std::string &prefix,
                    mfem::Mesh &mesh,
                    mfem::GridFunction &temperature,
                    mfem::GridFunction *angular_avg = nullptr,
                    mfem::GridFunction *qx = nullptr,
                    mfem::GridFunction *qy = nullptr,
                    mfem::GridFunction *qz = nullptr,
                    int order = 1,
                    bool high_order = true)
{
    mfem::ParaViewDataCollection pvd(prefix, &mesh);
    pvd.SetLevelsOfDetail(order);
    pvd.SetHighOrderOutput(high_order);
    pvd.SetPrecision(8);
    pvd.SetDataFormat(mfem::VTKFormat::BINARY);
    pvd.RegisterField("temperature", &temperature);
    if (angular_avg) { pvd.RegisterField("g_avg", angular_avg); }
    if (qx) { pvd.RegisterField("qx", qx); }
    if (qy) { pvd.RegisterField("qy", qy); }
    if (qz) { pvd.RegisterField("qz", qz); }
    pvd.SetCycle(0);
    pvd.SetTime(0.0);
    pvd.Save();
}

} // namespace dg

