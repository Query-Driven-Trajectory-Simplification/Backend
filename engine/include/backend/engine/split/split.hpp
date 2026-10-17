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

enum SplitType {
    Random,
    Time
};

TrajectoryDbSplit split(
    const core::TrajectoryDb& trajectories,
    SplitType split_type,
    double query_percentage,
    std::uint64_t seed = 42
);

}