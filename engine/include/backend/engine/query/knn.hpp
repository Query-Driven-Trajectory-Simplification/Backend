#pragma once

#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/query/frechet_helper.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

namespace query {

inline constexpr double kKnnResolutionM = 1e-3;

// Continuous distances are rounded up to a fixed millimetre grid using the decision predicate.
// The grid makes ranking independent of the search bounds, candidate order and requested k.
// Discrete distances are unchanged. Use this distance for cached kNN rankings too.
double knn_distance(const core::TrajectoryView &a, const core::TrajectoryView &b,
                    FrechetVariant variant = FrechetVariant::Continuous);

inline bool knn_rank_less(double distance_a, std::size_t index_a, double distance_b,
                          std::size_t index_b) {
    return distance_a < distance_b || (distance_a == distance_b && index_a < index_b);
}

struct KnnQuery {
    std::size_t query_index;            // index into KnnWorkload::queries
    std::vector<std::size_t> reference; // knn in the uncompressed database, closest first
};

struct KnnConfig {
    std::size_t k = 5;
    std::size_t query_count = 10; // 0 = use every trajectory in the query pool
    double p = 0.5; // top-k tau penalty for a pair one list holds neither of, in [0, 1]
    FrechetVariant frechet = FrechetVariant::Continuous;
};

struct KnnWorkload {
    core::TrajectoryDb queries;
    std::vector<KnnQuery> entries;
    std::size_t k;
    double p = 0.5;
    FrechetVariant frechet = FrechetVariant::Continuous;
    // query_index is a database trajectory, searched in its compressed form and excluded from its
    // own answer and reference. queries stays empty
    bool self = false;
};

// Samples query_count trajectories from query_pool (the held-out split) and stores their k
// nearest neighbours in original as the reference answer.
KnnWorkload generate_knn_queries(const core::TrajectoryDb &original,
                                 const core::TrajectoryDb &query_pool, const KnnConfig &config = {},
                                 std::uint64_t seed = 42);

// Indices into trajectories of the k trajectories closest to query by Fréchet distance, closest
// first, using knn_distance and knn_rank_less. Continuous distances in the same millimetre bin
// are tied by database index; discrete distance ties also use the database index.
std::vector<std::size_t> knn(const core::TrajectoryDb &trajectories,
                             const core::TrajectoryView &query, std::size_t k,
                             FrechetVariant variant = FrechetVariant::Continuous);
// A distance to database trajectory j already known, as knn_distance would give it, or nullopt.
// Fréchet distance is symmetric, so a search can reuse one found from the other side
using KnownDistance = std::function<std::optional<double>(std::size_t j)>;

// The same neighbours with the distances the search ranked them by: knn_distance for continuous
// Fréchet, the exact discrete distance otherwise
std::vector<std::pair<std::size_t, double>>
knn_with_distances(const core::TrajectoryDb &trajectories, const core::TrajectoryView &query,
                   std::size_t k, FrechetVariant variant = FrechetVariant::Continuous,
                   const KnownDistance &known = {});

} // namespace query
