#pragma once

#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/query/frechet_helper.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace query {

enum class JoinMode { HeldOut, Self };

struct JoinConfig {
    std::size_t query_count = 10;      // 0 = use every trajectory in the query pool
    double eps_m = 500.0;              // join threshold in metres, must be positive
    JoinMode mode = JoinMode::HeldOut; // Self ignores query_count and the query pool
    FrechetVariant frechet = FrechetVariant::Continuous;
};

struct JoinQuery {
    std::size_t query_index;
    std::vector<std::size_t> reference;
    std::vector<std::size_t> candidates;
};

struct JoinWorkload {
    // In Self mode this stores original endpoints/IDs for validation, not query inputs.
    core::TrajectoryDb queries;
    std::vector<JoinQuery> entries;
    double eps_m = 0.0;
    JoinMode mode = JoinMode::HeldOut;
    FrechetVariant frechet = FrechetVariant::Continuous;
    // Self entries contain only indices j > i; both operands are evaluated compressed.
};

// Samples query_count trajectories from query_pool (the held-out split) and stores their join
// result in original as the reference answer. Self mode delegates to generate_self_join.
JoinWorkload generate_join_queries(const core::TrajectoryDb &original,
                                   const core::TrajectoryDb &query_pool,
                                   const JoinConfig &config = {}, std::uint64_t seed = 42);

// All unordered database pairs, endpoint-pruned, with reference truth on originals.
// No held-out queries or sampling. Nonempty trajectories must retain their endpoints/IDs/order.
JoinWorkload generate_self_join(const core::TrajectoryDb &original, double eps_m = 500.0,
                                FrechetVariant variant = FrechetVariant::Continuous);

std::vector<std::size_t> join(const core::TrajectoryDb &trajectories,
                              const core::TrajectoryView &query, double eps,
                              FrechetVariant variant = FrechetVariant::Continuous);

} // namespace query
