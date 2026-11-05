#pragma once

#include <cstddef>
#include <vector>

#include "backend/engine/core/trajectory_db.hpp"

namespace allocation {

// Returns one budget per trajectory, proportional to size^alpha. alpha = 1 is size-proportional,
// alpha = 0 gives every trajectory the same budget. Keeps at least start and end points and never
// more points than a trajectory has.
std::vector<std::size_t> proportional_budget(const core::TrajectoryDb &trajectories,
                                             std::size_t total_budget, double alpha = 1.0);

} // namespace allocation
