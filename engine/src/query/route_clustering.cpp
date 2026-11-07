#include "backend/engine/query/route_clustering.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace query {

TrajectoryEndpoints endpoints_of(const core::TrajectoryView &trajectory) {
    if (trajectory.size() == 0)
        throw std::invalid_argument("route clustering: empty trajectory");
    return {trajectory.trajectory_id, trajectory.x.front(), trajectory.y.front(),
            trajectory.x.back(), trajectory.y.back()};
}

RouteClusters cluster_routes(const core::TrajectoryDb &trajectories,
                             std::span<const TrajectoryPair> neighbours, std::size_t min_size) {
    if (min_size < 2)
        throw std::invalid_argument("route clustering: min_size must be at least 2");
    const std::size_t n = trajectories.size();
    std::vector<std::vector<std::size_t>> adjacency(n);
    for (const auto &[i, j] : neighbours) {
        if (i >= n || j >= n || i == j)
            throw std::invalid_argument("route clustering: invalid neighbour pair");
        adjacency[i].push_back(j);
        adjacency[j].push_back(i);
    }
    // A repeated pair must not count twice towards the degree
    for (auto &list : adjacency) {
        std::ranges::sort(list);
        list.erase(std::ranges::unique(list).begin(), list.end());
    }

    // Max-heap on (degree, then tie-breaks). Degrees only fall, so stale entries are re-pushed.
    using Key = std::tuple<std::size_t, std::int64_t, double, std::int64_t, std::size_t>;
    const auto key = [&](std::size_t i, std::size_t degree) {
        const auto trajectory = trajectories.trajectory(i);
        if (trajectory.size() == 0)
            throw std::invalid_argument("route clustering: empty trajectory");
        return Key{degree, -trajectory.t.front(), -trajectory.x.front(),
                   -static_cast<std::int64_t>(trajectory.trajectory_id), i};
    };
    std::priority_queue<Key> heap;
    for (std::size_t i = 0; i < n; ++i)
        heap.push(key(i, adjacency[i].size()));

    RouteClusters clusters;
    clusters.label.assign(n, -1);
    while (!heap.empty()) {
        const auto [degree, neg_t, neg_x, neg_id, i] = heap.top();
        heap.pop();
        if (clusters.label[i] >= 0)
            continue;
        std::size_t free = 0;
        for (const auto j : adjacency[i])
            free += clusters.label[j] < 0;
        if (free < degree) {
            heap.push(key(i, free));
            continue;
        }
        if (free + 1 < min_size)
            break;
        const auto id = static_cast<std::ptrdiff_t>(clusters.members.size());
        auto &members = clusters.members.emplace_back();
        members.push_back(i);
        clusters.label[i] = id;
        for (const auto j : adjacency[i]) {
            if (clusters.label[j] < 0) {
                clusters.label[j] = id;
                members.push_back(j);
            }
        }
    }
    return clusters;
}

RouteClusteringWorkload generate_route_clustering(const core::TrajectoryDb &original,
                                                  const RouteClusteringConfig &config,
                                                  const JoinWorkload *self_join) {
    if (!std::isfinite(config.eps_m) || config.eps_m <= 0.0)
        throw std::invalid_argument("route clustering: eps_m must be finite and positive");

    RouteClusteringWorkload workload;
    workload.eps_m = config.eps_m;
    workload.min_size = config.min_size;
    workload.frechet = config.frechet;
    workload.originals.reserve(original.size());
    for (std::size_t i = 0; i < original.size(); ++i)
        workload.originals.push_back(endpoints_of(original.trajectory(i)));

    // A self join stores its database's IDs and endpoints, which must be these trajectories'
    const auto same_database = [&] {
        if (self_join->queries.size() != original.size() ||
            self_join->entries.size() != original.size())
            return false;
        for (std::size_t i = 0; i < original.size(); ++i) {
            const auto stored = self_join->queries.trajectory(i);
            if (stored.size() == 0 || endpoints_of(stored) != workload.originals[i])
                return false;
        }
        return true;
    };
    std::vector<TrajectoryPair> reference;
    const bool reuse = self_join && self_join->mode == JoinMode::Self &&
                       self_join->eps_m == config.eps_m && self_join->frechet == config.frechet &&
                       same_database();
    if (reuse) {
        for (const auto &entry : self_join->entries) {
            for (const auto j : entry.candidates)
                workload.candidates.emplace_back(entry.query_index, j);
            for (const auto j : entry.reference)
                reference.emplace_back(entry.query_index, j);
        }
    } else {
        const auto join = generate_self_join(original, config.eps_m, config.frechet);
        return generate_route_clustering(original, config, &join);
    }
    workload.reference = cluster_routes(original, reference, config.min_size);
    return workload;
}

} // namespace query
