#pragma once

#include "backend/engine/core/trajectory_db.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace query {

struct JoinConfig {
    std::size_t query_count = 10; // 0 = use every trajectory in the query pool
    double eps_m = 500.0;         // join threshold in metres, must be positive
};

struct JoinQuery {
    std::size_t query_index;
    std::vector<std::size_t> reference;
    std::vector<std::size_t> candidates;
};

struct JoinWorkload {
    core::TrajectoryDb queries;
    std::vector<JoinQuery> entries;
    double eps_m = 0.0;
};

// Samples query_count trajectories from query_pool (the held-out split) and stores their join
// result in original as the reference answer
JoinWorkload generate_join_queries(const core::TrajectoryDb &original,
                                   const core::TrajectoryDb &query_pool,
                                   const JoinConfig &config = {}, std::uint64_t seed = 42);

std::vector<std::size_t> join(const core::TrajectoryDb &trajectories,
                              const core::TrajectoryView &query, double eps);

} // namespace query
