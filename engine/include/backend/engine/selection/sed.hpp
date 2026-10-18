#pragma once

#include "backend/engine/core/trajectory_db.hpp"

#include <cstddef>
#include <vector>

namespace selection::sed {

// Synchronized Euclidean distance (SED), see https://arxiv.org/pdf/1801.05360
double synchronous_distance(const core::TrajectoryView &trajectory, std::size_t first,
                            std::size_t last, std::size_t point);

// Bottom-up greedy min-max SED simplification with a fixed point budget.
// Endpoints are always kept. Heuristic, not optimal.
std::vector<std::size_t> simplify(const core::TrajectoryView &trajectory,
                                  std::size_t points_to_keep);

} // namespace selection::sed
