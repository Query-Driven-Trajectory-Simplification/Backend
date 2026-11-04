#include "backend/engine/query/join.hpp"
#include "backend/engine/core/parallel.hpp"
#include "backend/engine/query/frechet_helper.hpp"
#include "backend/engine/query/knn.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace query {

namespace {

std::vector<std::size_t> endpoint_candidates(const core::TrajectoryDb &trajectories,
                                             const core::TrajectoryView &query, double eps) {
    std::vector<std::size_t> candidates;
    for (std::size_t i = 0; i < trajectories.size(); ++i) {
        if (endpoint_lower_bound(query, trajectories.trajectory(i)) <= eps)
            candidates.push_back(i);
    }
    return candidates;
}

double median(std::vector<double> values) {
    const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::ranges::nth_element(values, middle);
    return *middle;
}

} // namespace

std::vector<std::size_t> join(const core::TrajectoryDb &trajectories,
                              const core::TrajectoryView &query, double eps) {
    std::vector<std::size_t> result;
    for (const auto i : endpoint_candidates(trajectories, query, eps)) {
        if (frechet_within(query, trajectories.trajectory(i), eps))
            result.push_back(i);
    }
    return result;
}

JoinWorkload generate_join_queries(const core::TrajectoryDb &original,
                                   const core::TrajectoryDb &query_pool, const JoinConfig &config,
                                   std::uint64_t seed) {
    if (!std::isfinite(config.eps_m) || config.eps_m < 0.0)
        throw std::invalid_argument("join: eps_m must be finite and non-negative");
    if (config.eps_m == 0.0 && config.eps_neighbor == 0)
        throw std::invalid_argument("join: eps_neighbor must be positive when eps_m is 0");

    std::vector<std::size_t> picked(query_pool.size());
    std::iota(picked.begin(), picked.end(), 0);
    if (config.query_count != 0 && config.query_count < picked.size()) {
        std::mt19937_64 rng(seed);
        std::ranges::shuffle(picked, rng);
        picked.resize(config.query_count);
    }

    JoinWorkload results{core::subset(query_pool, picked), {}, config.eps_m};
    const std::size_t n = results.queries.size();
    if (n == 0 || original.size() == 0)
        return results;

    if (results.eps_m == 0.0) {
        const std::size_t rank = std::min(config.eps_neighbor, original.size());
        std::vector<double> distances(n);
        core::parallel_for(n, [&](std::size_t i) {
            const auto query = results.queries.trajectory(i);
            const auto neighbors = knn(original, query, rank);
            distances[i] = frechet_distance(query, original.trajectory(neighbors.back()));
        });
        results.eps_m = median(std::move(distances));
    }

    results.entries.resize(n);
    core::parallel_for(n, [&](std::size_t i) {
        const auto query = results.queries.trajectory(i);
        auto &entry = results.entries[i];
        entry.query_index = i;
        entry.candidates = endpoint_candidates(original, query, results.eps_m);
        for (const auto j : entry.candidates) {
            if (frechet_within(query, original.trajectory(j), results.eps_m))
                entry.reference.push_back(j);
        }
    });
    return results;
}

} // namespace query
