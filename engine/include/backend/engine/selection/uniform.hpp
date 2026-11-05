#pragma once

#include "backend/engine/core/trajectory_db.hpp"

#include <cstddef>
#include <vector>

namespace selection::uniform {

std::vector<std::size_t> simplify(const core::TrajectoryView &trajectory,
                                  std::size_t points_to_keep);

} // namespace selection::uniform
