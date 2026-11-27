#include "backend/engine/student/candidate_pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

namespace student {

std::vector<double> compute_arc_length(const core::TrajectoryView &traj) {
    const std::size_t num_pts = traj.size();
    std::vector<double> arc_length(num_pts, 0.0);
    for (std::size_t pt_idx = 1; pt_idx < num_pts; ++pt_idx) {
        const double dx = traj.x[pt_idx] - traj.x[pt_idx - 1];
        const double dy = traj.y[pt_idx] - traj.y[pt_idx - 1];
        arc_length[pt_idx] = arc_length[pt_idx - 1] + std::hypot(dx, dy);
    }
    return arc_length;
}

std::vector<std::size_t> build_candidate_pool(const core::TrajectoryView &traj,
                                              const CandidatePoolConfig &cfg) {
    const std::size_t num_pts = traj.size();
    if (num_pts <= 2) {
        std::vector<std::size_t> all_indices(num_pts);
        for (std::size_t i = 0; i < num_pts; ++i)
            all_indices[i] = i;
        return all_indices;
    }

    const auto arc_length = compute_arc_length(traj);
    std::vector<std::uint8_t> is_kept(num_pts, 0);
    is_kept[0] = 1;
    is_kept[num_pts - 1] = 1;

    // 1. Recursive top-down SED (> sed_tol_m) + arc-length stride (> arc_stride_m)
    const double sed_tol_sq = cfg.sed_tol_m * cfg.sed_tol_m;
    std::vector<std::pair<std::size_t, std::size_t>> pending_segments;
    pending_segments.reserve(64);
    pending_segments.emplace_back(0, num_pts - 1);

    while (!pending_segments.empty()) {
        const auto [start_idx, end_idx] = pending_segments.back();
        pending_segments.pop_back();
        if (end_idx <= start_idx + 1)
            continue;

        const double t_start = static_cast<double>(traj.t[start_idx]);
        const double dt = static_cast<double>(traj.t[end_idx]) - t_start;
        const double inv_dt = dt > 0.0 ? 1.0 / dt : 0.0;
        const double x_start = traj.x[start_idx], y_start = traj.y[start_idx];
        const double seg_dx = traj.x[end_idx] - x_start, seg_dy = traj.y[end_idx] - y_start;

        double max_sed_sq = -1.0;
        std::size_t furthest_idx = start_idx + 1;
        for (std::size_t pt_idx = start_idx + 1; pt_idx < end_idx; ++pt_idx) {
            const double frac = (static_cast<double>(traj.t[pt_idx]) - t_start) * inv_dt;
            const double dx = x_start + frac * seg_dx - traj.x[pt_idx];
            const double dy = y_start + frac * seg_dy - traj.y[pt_idx];
            const double sed_sq = dx * dx + dy * dy;
            if (sed_sq > max_sed_sq) {
                max_sed_sq = sed_sq;
                furthest_idx = pt_idx;
            }
        }

        if (max_sed_sq > sed_tol_sq) {
            is_kept[furthest_idx] = 1;
            pending_segments.emplace_back(start_idx, furthest_idx);
            pending_segments.emplace_back(furthest_idx, end_idx);
        } else if (cfg.arc_stride_m > 0.0 &&
                   (arc_length[end_idx] - arc_length[start_idx]) > cfg.arc_stride_m) {
            const double target_arc = 0.5 * (arc_length[start_idx] + arc_length[end_idx]);
            auto it = std::lower_bound(
                arc_length.begin() + static_cast<std::ptrdiff_t>(start_idx + 1),
                arc_length.begin() + static_cast<std::ptrdiff_t>(end_idx), target_arc);
            std::size_t midpoint_idx = static_cast<std::size_t>(it - arc_length.begin());
            midpoint_idx = std::clamp(midpoint_idx, start_idx + 1, end_idx - 1);
            is_kept[midpoint_idx] = 1;
            pending_segments.emplace_back(start_idx, midpoint_idx);
            pending_segments.emplace_back(midpoint_idx, end_idx);
        }
    }

    // 2. Spatial cell-run midpoints (protects point-in-box Range coverage)
    if (cfg.cell_m > 0.0) {
        const double inv_cell = 1.0 / cfg.cell_m;
        auto cell_of = [&](std::size_t pt_idx) {
            return std::pair<std::int64_t, std::int64_t>{
                static_cast<std::int64_t>(std::floor(traj.x[pt_idx] * inv_cell)),
                static_cast<std::int64_t>(std::floor(traj.y[pt_idx] * inv_cell)),
            };
        };
        std::size_t run_start = 0;
        auto prev_cell = cell_of(0);
        for (std::size_t pt_idx = 1; pt_idx < num_pts; ++pt_idx) {
            const auto cur_cell = cell_of(pt_idx);
            if (cur_cell != prev_cell) {
                is_kept[(run_start + pt_idx - 1) / 2] = 1;
                run_start = pt_idx;
                prev_cell = cur_cell;
            }
        }
        is_kept[(run_start + num_pts - 1) / 2] = 1;
    }

    std::vector<std::size_t> pool;
    pool.reserve(num_pts / 4 + 8);
    for (std::size_t pt_idx = 0; pt_idx < num_pts; ++pt_idx)
        if (is_kept[pt_idx])
            pool.push_back(pt_idx);
    return pool;
}

} // namespace student
