#pragma once

#include "backend/engine/core/trajectory_db.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace query {

struct JoinConfig {
    std::size_t query_count = 10; // 0 = use every trajectory in the query pool
    double eps_m = 0.0;           // join threshold in metres, 0 = derive it from the data
    // With eps_m = 0, eps is the median over the queries of the Fréchet distance to their
    // eps_neighbor-th nearest database trajectory, so a typical query has that many matches
    std::size_t eps_neighbor = 10;
};

struct JoinQuery {
    std::size_t query_index;             // index into JoinWorkload::queries
    std::vector<std::size_t> reference;  // database trajectories within eps, sorted
    std::vector<std::size_t> candidates; // within eps by the endpoint bound, sorted
};

// Similarity join between full-resolution query trajectories and the database: every database
// trajectory within discrete Fréchet distance eps of a query
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

// Indices into trajectories of those within discrete Fréchet distance eps of query, sorted
std::vector<std::size_t> join(const core::TrajectoryDb &trajectories,
                              const core::TrajectoryView &query, double eps);

} // namespace query
