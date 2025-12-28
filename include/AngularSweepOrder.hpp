// Compute element sweep order for each angular direction.
#pragma once

#include "AngularQuadrature.hpp"
#include "mfem.hpp"

#include <utility>
#include <vector>
#include <string>

namespace pbte
{
/// For each angular direction, stores an ordering of elements (indices into
/// mfem::Mesh) that is compatible with upwind traversal.
class AngularSweepOrder
{
public:
    /// Build sweep orders for all directions in angle_quad on the given mesh.
    /// Uses face-based precedence: if dir · n_out < 0 for a face shared with a
    /// neighbor, the neighbor must appear earlier in the order.
    static AngularSweepOrder Build(const mfem::Mesh &mesh,
                                   const AngleQuadrature &angle_quad);

    /// Number of directions (matches angle_quad.Directions()).
    int NumDirections() const { return static_cast<int>(orders_.size()); }

    /// Ordering for direction k (size == mesh.GetNE()).
    const std::vector<int> &Order(int k) const { return orders_[k]; }

    /// All orders.
    const std::vector<std::vector<int>> &Orders() const { return orders_; }

    /// Write summary and per-direction orders to a file path.
    void WriteToFile(const AngleQuadrature &angle_quad,
                     const mfem::Mesh &mesh,
                     const std::string &path) const;

private:
    std::vector<std::vector<int>> orders_;
};
}  // namespace pbte


