#pragma once

#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/query/join.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace query {

struct RouteClusteringConfig {
    bool enabled = false;
    double eps_m = 500.0;     // neighbour threshold, same test as the self join
    std::size_t min_size = 5; // a cluster needs at least this many trajectories, medoid included
    FrechetVariant frechet = FrechetVariant::Continuous;
};

using TrajectoryPair = std::pair<std::size_t, std::size_t>;

// Greedy medoid grouping. Repeatedly takes the unassigned trajectory with the most unassigned
// neighbours and groups it with them if that gives at least min_size trajectories. Every member
// neighbours its medoid, so clusters cannot chain. Ties go to the earliest start time, then the
// smallest start x, then the smallest trajectory ID, which makes the result order independent.
struct RouteClusters {
    std::vector<std::ptrdiff_t> label;             // cluster per trajectory, -1 = unclustered
    std::vector<std::vector<std::size_t>> members; // members[c][0] is the medoid
};

RouteClusters cluster_routes(const core::TrajectoryDb &trajectories,
                             std::span<const TrajectoryPair> neighbours, std::size_t min_size);

// ID and endpoints of an original trajectory. The candidate pairs were pruned by endpoint
// distance, so a database is only comparable if every one of these matches.
struct TrajectoryEndpoints {
    std::int32_t trajectory_id = 0;
    double first_x = 0.0, first_y = 0.0, last_x = 0.0, last_y = 0.0;
    bool operator==(const TrajectoryEndpoints &) const = default;
};

TrajectoryEndpoints endpoints_of(const core::TrajectoryView &trajectory);

struct RouteClusteringWorkload {
    std::vector<TrajectoryPair> candidates; // endpoint-pruned pairs, i < j
    RouteClusters reference;
    std::vector<TrajectoryEndpoints> originals;
    double eps_m = 0.0;
    std::size_t min_size = 0;
    FrechetVariant frechet = FrechetVariant::Continuous;
};

// Clusters the original database. A self join workload over the same trajectories, eps and
// Fréchet variant is reused instead of recomputing the Fréchet tests on the originals.
RouteClusteringWorkload generate_route_clustering(const core::TrajectoryDb &original,
                                                  const RouteClusteringConfig &config = {},
                                                  const JoinWorkload *self_join = nullptr);

} // namespace query
