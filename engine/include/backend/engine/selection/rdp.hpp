#pragma once

#include "backend/engine/core/trajectory_db.hpp"

#include <cstddef>
#include <vector>

namespace selection::rdp {

// Top-down Ramer-Douglas-Peucker (RDP) simplification adapted for a fixed point budget.
std::vector<std::size_t> simplify(const core::TrajectoryView &trajectory,
                                  std::size_t points_to_keep);

// Endpoints first, then points in importance order. Sorting the first b entries
// reproduces simplify(trajectory, b), for 2 <= b <= min(limit, size).
std::vector<std::size_t> sequence(const core::TrajectoryView &trajectory, std::size_t limit);

} // namespace selection::rdp
