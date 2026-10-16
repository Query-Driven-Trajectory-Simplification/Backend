#pragma once

#include "backend/engine/core/trajectory_db.hpp"
#include <cstddef>
#include <vector>
namespace allocation {

enum class AlgorithmType { kSizeProportional };

std::vector<std::size_t> create_trajectory_budgets(const core::TrajectoryDb &trajectories,
                                                   AlgorithmType algorithm,
                                                   std::size_t total_budget);
} // namespace allocation
