#pragma once

#include "backend/engine/core/trajectory_db.hpp"
#include <vector>
#include <cstdint>
#include <cstddef>

namespace selection {

enum class AlgorithmType {
    RDP,
    SED
};


std::vector<std::int64_t> simplify_trajectory(
    const core::TrajectoryView&, 
    AlgorithmType,
    std::size_t);

}