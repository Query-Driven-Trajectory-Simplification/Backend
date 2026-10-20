#pragma once

#include "backend/engine/core/trajectory_db.hpp"

#include <cstddef>
#include <vector>

namespace query {

// Indices into trajectories of the k trajectories closest to query, closest first.
// Distance measure not decided yet, so this is a placeholder that throws.
std::vector<std::size_t> knn(const core::TrajectoryDb &trajectories,
                             const core::TrajectoryView &query, std::size_t k);

} // namespace query
