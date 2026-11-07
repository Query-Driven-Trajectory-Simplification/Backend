#include "backend/engine/query/join.hpp"
#include "backend/engine/core/parallel.hpp"
#include "backend/engine/core/simplification.hpp"
#include "backend/engine/query/frechet_helper.hpp"

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

} // namespace

std::vector<std::size_t> join(const core::TrajectoryDb &trajectories,
                              const core::TrajectoryView &query, double eps,
                              FrechetVariant variant) {
    std::vector<std::size_t> result;
    for (const auto i : endpoint_candidates(trajectories, query, eps)) {
        if (frechet_within(query, trajectories.trajectory(i), eps, variant))
            result.push_back(i);
    }
    return result;
}

JoinWorkload generate_join_queries(const core::TrajectoryDb &original,
                                   const core::TrajectoryDb &query_pool, const JoinConfig &config,
                                   std::uint64_t seed) {
    if (!std::isfinite(config.eps_m) || config.eps_m <= 0.0)
        throw std::invalid_argument("join: eps_m must be finite and positive");

    if (config.mode == JoinMode::Self)
        return generate_self_join(original, config.eps_m, config.frechet);

    std::vector<std::size_t> picked(query_pool.size());
    std::iota(picked.begin(), picked.end(), 0);
    if (config.query_count != 0 && config.query_count < picked.size()) {
        std::mt19937_64 rng(seed);
        std::ranges::shuffle(picked, rng);
        picked.resize(config.query_count);
    }

    JoinWorkload results{
        core::subset(query_pool, picked), {}, config.eps_m, JoinMode::HeldOut, config.frechet};
    const std::size_t n = results.queries.size();
    if (n == 0 || original.size() == 0)
        return results;

    results.entries.resize(n);
    core::parallel_for(n, [&](std::size_t i) {
        const auto query = results.queries.trajectory(i);
        auto &entry = results.entries[i];
        entry.query_index = i;
        entry.candidates = endpoint_candidates(original, query, results.eps_m);
        for (const auto j : entry.candidates) {
            if (frechet_within(query, original.trajectory(j), results.eps_m, config.frechet))
                entry.reference.push_back(j);
        }
    });
    return results;
}

JoinWorkload generate_self_join(const core::TrajectoryDb &original, double eps_m,
                                FrechetVariant variant) {
    if (!std::isfinite(eps_m) || eps_m <= 0.0)
        throw std::invalid_argument("self join: eps_m must be finite and positive");
    core::Simplification endpoints;
    for (std::size_t i = 0; i < original.size(); ++i) {
        const auto n = original.trajectory(i).size();
        if (n == 0)
            throw std::invalid_argument("self join: empty trajectory");
        const std::vector<std::size_t> kept =
            n == 1 ? std::vector<std::size_t>{0} : std::vector<std::size_t>{0, n - 1};
        endpoints.add_trajectory(kept);
    }
    JoinWorkload result{core::materialize(original, endpoints), {}, eps_m, JoinMode::Self, variant};
    result.entries.resize(original.size());
    core::parallel_for(original.size(), [&](std::size_t i) {
        auto &entry = result.entries[i];
        entry.query_index = i;
        const auto a = original.trajectory(i);
        for (std::size_t j = i + 1; j < original.size(); ++j) {
            const auto b = original.trajectory(j);
            if (endpoint_lower_bound(a, b) > eps_m)
                continue;
            entry.candidates.push_back(j);
            if (frechet_within(a, b, eps_m, variant))
                entry.reference.push_back(j);
        }
    });
    return result;
}

} // namespace query
