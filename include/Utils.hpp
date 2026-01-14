#pragma once

#include "mfem.hpp"

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
}  // namespace utils
