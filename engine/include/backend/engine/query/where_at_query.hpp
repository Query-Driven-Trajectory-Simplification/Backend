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
    std::uint64_t seed = 1;
    double max_gap_seconds = 60.0;
};

struct WhereAtWorkload {
    std::vector<WhereAtQuery> queries;
    std::size_t skipped_trajectories = 0;
};

WhereAtWorkload generate_where_at_queries(const core::TrajectoryDb &original,
                                          const WhereAtConfig &config = {});

} // namespace query
