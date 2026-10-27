#pragma once

#include "backend/engine/core/local_point.hpp"
#include "backend/engine/core/trajectory_db.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace query {

struct WhereAtQuery {
    std::size_t trajectory_index;
    std::int32_t trajectory_id;
    double t;
    core::LocalPoint reference;
};

struct WhereAtConfig {
    std::size_t probes_per_trajectory = 128;
    double max_gap_seconds = 60.0;
};

struct WhereAtWorkload {
    std::vector<WhereAtQuery> queries;
    std::size_t skipped_trajectories = 0;
};

// Generates probes_per_trajectory queries for each trajectory with eligible time.
// Validates WhereAtConfig, then finds intervals between consecutive points whose
// time gap is at most max_gap_seconds.
//
// Longer gaps are excluded because linear interpolation may poorly represent
// the actual movement. For example, observations in Skagen at 12:00 and Aarhus
// at 15:00 do not tell us where the traveller was at 13:30. Interpolation would
// place them halfway along a straight line, potentially far from their route.
//
// Samples query times across the eligible intervals and stores the original
// interpolated position at each time as the reference answer.
WhereAtWorkload generate_where_at_queries(const core::TrajectoryDb &original,
                                          const WhereAtConfig &config = {},
                                          std::uint64_t seed = 42);

} // namespace query
