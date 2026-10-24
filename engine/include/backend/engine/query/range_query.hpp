#pragma once

#include "backend/engine/core/trajectory_db.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace query {

// Axis-aligned box in local x/y (meters) and a closed time window [t_min, t_max].
struct RangeBox {
    double x_min;
    double x_max;
    double y_min;
    double y_max;
    std::int64_t t_min;
    std::int64_t t_max;
};

struct RangeQuery {
    RangeBox box;
    std::vector<std::size_t> reference; // trajectories in the uncompressed database inside box
};

struct RangeConfig {
    std::size_t query_count = 20000;
    double box_size_m = 2000.0;
    std::int64_t window_seconds = 604800; // 7 days
    std::size_t min_results = 5;
    std::size_t full_accept_at = 20;
    std::size_t attempts_per_query = 1000;
    std::uint64_t seed = 42;
};

struct RangeWorkload {
    std::vector<RangeQuery> queries;
};

RangeWorkload generate_range_queries(const core::TrajectoryDb &original,
                                     const RangeConfig &config = {});

std::vector<std::size_t> range(const core::TrajectoryDb &trajectories, const RangeBox &box);

} // namespace query
