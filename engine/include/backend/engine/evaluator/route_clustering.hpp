#pragma once

#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/query/route_clustering.hpp"

#include <cstddef>
#include <limits>

namespace evaluator {

// Reruns the route clustering on the compressed database and compares it with the reference.
// score is the symmetric macro best-match F1: each cluster's best overlap F1 with any cluster on
// the other side, averaged per side, then over both sides. NaN without reference clusters, 0 if
// nothing is predicted.
struct RouteClusteringScores {
    static constexpr double missing = std::numeric_limits<double>::quiet_NaN();

    double score = missing;
    std::size_t reference_clusters = 0;
    std::size_t predicted_clusters = 0;
    std::size_t reference_covered = 0;
    std::size_t predicted_covered = 0;
};

query::RouteClusters compressed_route_clusters(const core::TrajectoryDb &compressed,
                                               const query::RouteClusteringWorkload &workload);

RouteClusteringScores compare_route_clusters(const query::RouteClusters &reference,
                                             const query::RouteClusters &predicted);

RouteClusteringScores evaluate_route_clustering(const core::TrajectoryDb &compressed,
                                                const query::RouteClusteringWorkload &workload);

} // namespace evaluator
