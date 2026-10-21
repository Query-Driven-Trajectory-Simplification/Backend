#pragma once

#include "backend/engine/core/simplification.hpp"
#include "backend/engine/core/trajectory_db.hpp"
#include <cstddef>
#include <span>
#include <vector>

namespace selection {

enum class AlgorithmType { RDP, SED };

std::vector<std::size_t> simplify_trajectory(const core::TrajectoryView &, AlgorithmType,
                                             std::size_t);

// Simplifies every trajectory, trajectory i keeping budgets[i] points
core::Simplification simplify_database(const core::TrajectoryDb &trajectories,
                                       AlgorithmType algorithm,
                                       std::span<const std::size_t> budgets);

} // namespace selection
