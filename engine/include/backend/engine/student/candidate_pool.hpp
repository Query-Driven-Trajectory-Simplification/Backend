#pragma once

#include "backend/engine/core/trajectory_db.hpp"

#include <cstddef>
#include <vector>

namespace student {

struct CandidatePoolConfig {
    double sed_tol_m = 5.0;
    double arc_stride_m = 250.0;
    double cell_m = 1000.0;
};

// Cumulative arc length in meters along a trajectory.
std::vector<double> compute_arc_length(const core::TrajectoryView &traj);

// Fast safe candidate pool (~4x reduction, >99.5% Teacher J4 ceiling):
// union of top-down SED (> sed_tol_m), arc-length stride (<= arc_stride_m),
// and spatial cell-run midpoints (cell_m). Sorted, endpoints always included.
std::vector<std::size_t> build_candidate_pool(const core::TrajectoryView &traj,
                                              const CandidatePoolConfig &cfg = {});

} // namespace student
