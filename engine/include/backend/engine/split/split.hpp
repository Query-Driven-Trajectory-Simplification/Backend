#pragma once

#include "backend/engine/core/trajectory_db.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace split {

struct TrajectoryDbSplit {
    std::vector<std::size_t> database;
    std::vector<std::size_t> queries;
};

enum class SplitType { Random, Time };

struct SplitConfig {
    SplitType type = SplitType::Random;
    double query_percentage = 0.15;
    std::uint64_t seed = 1; // only used by the random split
};

TrajectoryDbSplit split(const core::TrajectoryDb &trajectories, const SplitConfig &config);

} // namespace split
