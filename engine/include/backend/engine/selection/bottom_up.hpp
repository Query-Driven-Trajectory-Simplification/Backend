#pragma once

#include "backend/engine/core/trajectory_db.hpp"

#include <cstddef>
#include <vector>

namespace selection::bottom_up {

// Bottom-up greedy min-max simplification with a fixed point budget: start from every point and
// repeatedly drop the interior point whose removal gives the smallest span error, the worst
// error of the original points between its kept neighbours. Endpoints are always kept.
// The measures follow the error-driven baselines of the trajectory simplification benchmark
// (Wang et al., PVLDB 2021) as MLSimp's evaluation code defines them
enum class Measure {
    Sed, // synchronized Euclidean distance: the point against the segment position at its time
    Ped, // perpendicular distance to the line through the kept neighbours, 0 if they coincide
    Dad, // direction-aware distance: largest angle between an original segment's heading and
         // the kept segment's heading, in radians
};

// Worst error of the original points strictly between first and last (Ped, Sed), or of the
// original segments from first to last (Dad), against the segment first -> last. Metres for Sed
// and Ped, radians for Dad. Requires first < last
double span_error(const core::TrajectoryView &trajectory, std::size_t first, std::size_t last,
                  Measure measure);

std::vector<std::size_t> simplify(const core::TrajectoryView &trajectory,
                                  std::size_t points_to_keep, Measure measure);

// Endpoints first, then points in reverse removal order. Sorting the first b entries
// reproduces simplify(trajectory, b, measure), for 2 <= b <= min(limit, size)
std::vector<std::size_t> sequence(const core::TrajectoryView &trajectory, std::size_t limit,
                                  Measure measure);

} // namespace selection::bottom_up
