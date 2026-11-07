#include "backend/engine/evaluator/route_clustering.hpp"
#include "backend/engine/core/parallel.hpp"
#include "backend/engine/query/frechet_helper.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace evaluator {

namespace {

// Mean over a's clusters of the best overlap F1 with any of b's clusters
double best_match(const query::RouteClusters &a, const query::RouteClusters &b) {
    double sum = 0.0;
    for (const auto &members : a.members) {
        std::unordered_map<std::ptrdiff_t, std::size_t> overlap;
        for (const auto i : members) {
            if (b.label[i] >= 0)
                ++overlap[b.label[i]];
        }
        double best = 0.0;
        for (const auto &[cluster, shared] : overlap) {
            const auto size = members.size() + b.members[cluster].size();
            best = std::max(best, 2.0 * static_cast<double>(shared) / static_cast<double>(size));
        }
        sum += best;
    }
    return sum / static_cast<double>(a.members.size());
}

std::size_t covered(const query::RouteClusters &clusters) {
    std::size_t total = 0;
    for (const auto &members : clusters.members)
        total += members.size();
    return total;
}

} // namespace

query::RouteClusters compressed_route_clusters(const core::TrajectoryDb &compressed,
                                               const query::RouteClusteringWorkload &workload) {
    if (compressed.size() != workload.originals.size())
        throw std::invalid_argument("route clustering: database size does not match reference");
    // Simplification keeps endpoints, so the original candidates cover every possible match
    for (std::size_t i = 0; i < compressed.size(); ++i) {
        const auto trajectory = compressed.trajectory(i);
        if (trajectory.size() == 0 || query::endpoints_of(trajectory) != workload.originals[i])
            throw std::invalid_argument(
                "route clustering: IDs/order or endpoints do not match reference");
    }

    std::vector<char> matched(workload.candidates.size(), 0);
    core::parallel_for(workload.candidates.size(), [&](std::size_t k) {
        const auto [i, j] = workload.candidates[k];
        matched[k] = query::frechet_within(compressed.trajectory(i), compressed.trajectory(j),
                                           workload.eps_m);
    });
    std::vector<query::TrajectoryPair> neighbours;
    for (std::size_t k = 0; k < matched.size(); ++k) {
        if (matched[k])
            neighbours.push_back(workload.candidates[k]);
    }
    return query::cluster_routes(compressed, neighbours, workload.min_size);
}

RouteClusteringScores compare_route_clusters(const query::RouteClusters &reference,
                                             const query::RouteClusters &predicted) {
    if (reference.label.size() != predicted.label.size())
        throw std::invalid_argument("route clustering: label sizes differ");

    RouteClusteringScores scores;
    scores.reference_clusters = reference.members.size();
    scores.predicted_clusters = predicted.members.size();
    scores.reference_covered = covered(reference);
    scores.predicted_covered = covered(predicted);
    if (reference.members.empty())
        return scores;

    scores.score =
        predicted.members.empty()
            ? 0.0
            : 0.5 * (best_match(reference, predicted) + best_match(predicted, reference));
    return scores;
}

RouteClusteringScores evaluate_route_clustering(const core::TrajectoryDb &compressed,
                                                const query::RouteClusteringWorkload &workload) {
    return compare_route_clusters(workload.reference,
                                  compressed_route_clusters(compressed, workload));
}

} // namespace evaluator
