#include "backend/engine/split/split.hpp"
#include "backend/engine/core/trajectory_db.hpp"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

void validate(double value) {
    if (value < 0.0 || value > 1.0) {
        throw std::invalid_argument("Value must be between 0 and 1");
    }
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

    const auto split_index = static_cast<std::size_t>(trajectories.size() * (1.0 - query_percentage));

    std::vector<std::size_t> train_indices(indices.begin(), indices.begin() + split_index);
    std::vector<std::size_t> query_indices(indices.begin() + split_index, indices.end());

    return split::TrajectoryDbSplit{
        .database = std::move(train_indices),
        .queries = std::move(query_indices)
    };
}

} // namespace

namespace split {

TrajectoryDbSplit split(const core::TrajectoryDb& trajectories, SplitType split_type, double query_percentage) {
    switch (split_type) {
        case SplitType::Random:
            throw std::runtime_error("Splitting randomly not implemented yet");
        case SplitType::Time:
            return time_split(trajectories, query_percentage);
    }

    throw std::invalid_argument("Unknown algo type");
}

} // namespace split