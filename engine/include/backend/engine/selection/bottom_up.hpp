#pragma once

#include "backend/engine/core/simplification.hpp"
#include "backend/engine/core/trajectory_db.hpp"

#include <cstddef>
#include <vector>

namespace selection::bottom_up {

enum class Measure {
    Sed, // synchronized Euclidean distance: the point against the segment position at its time
    Ped, // distance to the segment between the kept neighbours, as in RDP
    Dad, // direction-aware distance: largest angle between an original segment's heading and
         // the kept segment's heading, in radians
};

double span_error(const core::TrajectoryView &trajectory, std::size_t first, std::size_t last,
                  Measure measure);

std::vector<std::size_t> simplify(const core::TrajectoryView &trajectory,
                                  std::size_t points_to_keep, Measure measure);

std::vector<std::size_t> sequence(const core::TrajectoryView &trajectory, std::size_t limit,
                                  Measure measure);

core::Simplification simplify_database(const core::TrajectoryDb &db, std::size_t total_budget,
                                       Measure measure);

} // namespace selection::bottom_up
