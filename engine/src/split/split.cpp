#include "backend/engine/split/split.hpp"
#include "backend/engine/core/trajectory_db.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

void validate(double value) {
    if (!(value >= 0.0 && value <= 1.0)) {
        throw std::invalid_argument("Value must be between 0 and 1");
    }
}

std::size_t compute_split_index(std::size_t n, double query_percentage) {
    const auto query_count = static_cast<std::size_t>(std::round(static_cast<double>(n) * query_percentage));
    return n - query_count;
}

std::int64_t start_time(const core::TrajectoryDb& db, std::size_t idx) {
    const auto view = db.trajectory(idx);
    return view.t.empty() ? std::numeric_limits<std::int64_t>::max() : view.t.front();
}

split::TrajectoryDbSplit time_split(const core::TrajectoryDb& trajectories, double query_percentage) {
    validate(query_percentage);

    std::vector<std::size_t> indices(trajectories.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::stable_sort(indices.begin(), indices.end(), [&](std::size_t a, std::size_t b) {
        return start_time(trajectories, a) < start_time(trajectories, b);
    });

    const auto split_index = compute_split_index(trajectories.size(), query_percentage);

    return split::TrajectoryDbSplit{
        .database = {indices.begin(), indices.begin() + split_index},
        .queries  = {indices.begin() + split_index, indices.end()}
    };
}

split::TrajectoryDbSplit random_split(const core::TrajectoryDb& trajectories, double query_percentage, std::uint64_t seed) {
    validate(query_percentage);

    const std::size_t n = trajectories.size();
    const auto split_index = compute_split_index(n, query_percentage);
    std::size_t queries_needed = n - split_index;

    std::vector<std::size_t> train_indices;
    std::vector<std::size_t> query_indices;
    train_indices.reserve(split_index);
    query_indices.reserve(queries_needed);

    std::mt19937_64 rng(seed);

    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t remaining = n - i;

        if ((rng() % remaining) < queries_needed) {
            query_indices.push_back(i);
            --queries_needed;
        } else {
            train_indices.push_back(i);
        }
    }

    return split::TrajectoryDbSplit{
        .database = std::move(train_indices),
        .queries  = std::move(query_indices)
    };
}

} // namespace

namespace split {

TrajectoryDbSplit split(
    const core::TrajectoryDb& trajectories,
    SplitType split_type,
    double query_percentage,
    std::uint64_t seed
) {
    switch (split_type) {
        case SplitType::Random:
            return random_split(trajectories, query_percentage, seed);
        case SplitType::Time:
            return time_split(trajectories, query_percentage);
    }

    throw std::invalid_argument("Unknown split type");
}

} // namespace split