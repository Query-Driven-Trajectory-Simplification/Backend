#include "backend/engine/split/split.hpp"
#include "backend/engine/core/trajectory_db.hpp"
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

void validate(double value) {
    if (value < 0.0 || value > 1.0) {
        throw std::invalid_argument("Value must be between 0 and 1");
    }
}

split::TrajectoryDbSplit time_split(core::TrajectoryDb trajectories, double query_percentage) {
    validate(query_percentage);

    const auto query_count = static_cast<size_t>(std::round(trajectories.size() * query_percentage));
    const auto split_index = trajectories.size() - query_count;

    std::vector<size_t> train_indices(split_index);
    std::vector<size_t> query_indices(trajectories.size() - split_index);

    std::iota(train_indices.begin(), train_indices.end(), 0);
    std::iota(query_indices.begin(), query_indices.end(), split_index);

    return split::TrajectoryDbSplit{.database = train_indices, .queries = query_indices};
}

}

namespace split {

split::TrajectoryDbSplit split(core::TrajectoryDb trajectories, split::SplitType split_type, double query_percentage) {
    switch (split_type) {
        case split::Random:
            throw std::runtime_error("Spltting by time not implemented yet");
        case split::Time:
            return time_split(trajectories, query_percentage);
    }

    throw std::invalid_argument("Unknown algo type");
}

}