#pragma once

#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/student/density_index.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace student {

struct SubmodularWeights {
    double w_geo = 0.4; // perpendicular Shape (d_perp + L1 area + arc ramp) -> kNN & Join
    double w_wa = 1.3;  // 64-probe temporal Where-At 25-200m ramp -> Where-At
    double w_rng = 2.2; // submodular uncovered 333m x 3.5d cell coverage -> Range
    int rb_passes = 2;  // fast red-black free-knot polish passes
};

// Fast, zero-Frechet submodular greedy + red-black free-knot solver on the Safe Candidate Pool,
// using linear analytical Range Macro-F1 cell weights over populated 333m x 3.5d cells within
// +-500m x +-3.5d of the trajectory.
std::vector<std::size_t> submodular_simplify_on_pool(const core::TrajectoryView &traj,
                                                     std::span<const std::size_t> pool,
                                                     std::size_t budget,
                                                     const SubmodularWeights &w,
                                                     const DbDensityIndex &density_idx);

} // namespace student
