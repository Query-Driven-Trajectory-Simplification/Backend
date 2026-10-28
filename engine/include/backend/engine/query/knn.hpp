#pragma once

#include "backend/engine/core/trajectory_db.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace query {

struct KnnQuery {
    std::size_t query_index;            // index into KnnWorkload::queries
    std::vector<std::size_t> reference; // knn in the uncompressed database, closest first
};

struct KnnConfig {
    std::size_t k = 5;
    std::size_t query_count = 10; // 0 = use every trajectory in the query pool
};

struct KnnWorkload {
    core::TrajectoryDb queries;
    std::vector<KnnQuery> entries;
    std::size_t k;
};

// Samples query_count trajectories from query_pool (the held-out split) and stores their k
// nearest neighbours in original as the reference answer.
KnnWorkload generate_knn_queries(const core::TrajectoryDb &original,
                                 const core::TrajectoryDb &query_pool, const KnnConfig &config = {},
                                 std::uint64_t seed = 42);

// Indices into trajectories of the k trajectories closest to query by discrete Fréchet
// distance, closest first. Equal distances are ordered by database index.
std::vector<std::size_t> knn(const core::TrajectoryDb &trajectories,
                             const core::TrajectoryView &query, std::size_t k);

} // namespace query
