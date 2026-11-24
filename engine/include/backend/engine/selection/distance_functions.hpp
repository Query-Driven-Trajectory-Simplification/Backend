#pragma once

#include "backend/engine/core/trajectory_db.hpp"

#include <cstddef>
#include <vector>

namespace selection::distance {

double max_sed_squared(const core::TrajectoryView &trajectory, std::size_t first, std::size_t last);

double max_ped_squared(const core::TrajectoryView &trajectory, std::size_t first, std::size_t last);

// vector of direction from points k -> k +1, measured in [0, 2pi)
std::vector<double> segment_headings(const core::TrajectoryView &trajectory);

// Max circular angle, in radians, between the heading of first -> last and of each original segment
// from first to last
double max_heading_change(const core::TrajectoryView &trajectory,
                          const std::vector<double> &headings, std::size_t first, std::size_t last);

} // namespace selection::distance
