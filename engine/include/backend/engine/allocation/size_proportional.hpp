#pragma once

#include <cstddef>
#include <vector>

#include "backend/engine/core/trajectory_db.hpp"

namespace allocation {

// Returns vector of budgets for each trip, which is proportional to the number of points in the
// trip. Keeps at least start and end points
std::vector<std::size_t> proportional_budget(const core::TrajectoryDb &trajectories,
                                             std::size_t total_budget);

} // namespace allocation
