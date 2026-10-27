#pragma once

#include "backend/engine/core/trajectory_db.hpp"
#include <cstddef>
#include <vector>
namespace allocation {
enum class AllocatorType { SizeProportional };

std::vector<std::size_t> create_trajectory_budgets(const core::TrajectoryDb &trajectories,
                                                   AllocatorType allocator,
                                                   std::size_t total_budget);
} // namespace allocation
