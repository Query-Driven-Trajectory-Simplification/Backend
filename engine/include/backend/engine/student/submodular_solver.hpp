#pragma once

#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/student/density_index.hpp"
#include "backend/engine/student/range_sampler.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace student {

struct SubmodularWeights {
    double w_geo = 0.4; // perpendicular Shape (d_perp + L1 area + arc ramp) -> kNN & Join
    double w_wa = 1.3;  // 64-probe temporal Where-At 25-200m ramp -> Where-At
    double w_rng = 2.2; // submodular uncovered Range coverage -> Range
    int rb_passes = 2;  // fast red-black free-knot polish passes
};

// Fast, zero-Frechet submodular greedy + red-black free-knot solver on the Safe Candidate Pool.
// When uncov_boxes is non-null, blends 80% exact sampled 1km x 7d boxes + 20% 333m cell grid.
std::vector<std::size_t>
submodular_simplify_on_pool(const core::TrajectoryView &traj, std::span<const std::size_t> pool,
                            std::size_t budget, const SubmodularWeights &w,
                            const DbDensityIndex &density_idx,
                            const std::vector<SampledRangeBox> *uncov_boxes = nullptr,
                            double uncov_frac = 1.0);

} // namespace student
