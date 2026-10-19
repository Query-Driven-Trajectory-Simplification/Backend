#pragma once

#include "backend/engine/core/trajectory_db.hpp"
#include <cstddef>
#include <vector>

namespace selection {

enum class AlgorithmType { RDP, SED };

std::vector<std::size_t> simplify_trajectory(const core::TrajectoryView &, AlgorithmType,
                                             std::size_t);

} // namespace selection