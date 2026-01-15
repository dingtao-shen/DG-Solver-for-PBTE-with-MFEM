#pragma once

#include "mfem.hpp"
#include "AngularQuadrature.hpp"

#include <ostream>
#include <string>
#include <vector>

namespace pbte
{
struct ElementIntegralData;
}

namespace utils
{
/// Write vector contents with a label to a stream.
void WriteVector(std::ostream &os, const mfem::Vector &v,
                 const std::string &label);

/// Write dense matrix contents with a label to a stream.
void WriteMatrix(std::ostream &os, const mfem::DenseMatrix &m,
                 const std::string &label);

/// Serialize element integrals and write to output/log/integrals_all.txt.
/// Returns true on success (or non-root ranks in MPI).
bool DumpElementIntegrals(const std::vector<pbte::ElementIntegralData> &element_data,
                          bool is_root);

/// Dump coefficient blocks coeff[dir][branch][spec] to ./output/log as
/// per-direction/per-branch/per-spec files. Only root writes when MPI is enabled.
bool DumpCoefficients(const std::vector<std::vector<std::vector<mfem::DenseMatrix>>> &coeff,
                      const pbte::AngleQuadrature &quad,
                      bool is_root);

/// Dump macroscopic temperature Tc (ndof x ne) to ./output/log/Tc_all.txt.
/// Only root writes when MPI is enabled.
bool DumpTemperature(const mfem::DenseMatrix &Tc, bool is_root);

/// Compute a unit normal for a mesh face using its vertices.
mfem::Vector ComputeFaceNormal(const mfem::Mesh &mesh, int face_id);

/// Compute a face normal oriented outward from the given element.
mfem::Vector ComputeOutwardFaceNormal(const mfem::Mesh &mesh,
                                      int face_id,
                                      int elem_id);
}  // namespace utils
