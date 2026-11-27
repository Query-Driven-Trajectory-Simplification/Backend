#include "backend/engine/student/candidate_pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

namespace student {

std::vector<double> compute_arc_length(const core::TrajectoryView &traj) {
    const std::size_t n = traj.size();
    std::vector<double> arc(n, 0.0);
    for (std::size_t k = 1; k < n; ++k) {
        const double dx = traj.x[k] - traj.x[k - 1];
        const double dy = traj.y[k] - traj.y[k - 1];
        arc[k] = arc[k - 1] + std::hypot(dx, dy);
    }
    return arc;
}

std::vector<std::size_t> build_candidate_pool(const core::TrajectoryView &traj,
                                              const CandidatePoolConfig &cfg) {
    const std::size_t n = traj.size();
    if (n <= 2) {
        std::vector<std::size_t> out(n);
        for (std::size_t i = 0; i < n; ++i)
            out[i] = i;
        return out;
    }

    const auto arc = compute_arc_length(traj);
    std::vector<std::uint8_t> keep(n, 0);
    keep[0] = 1;
    keep[n - 1] = 1;

    // 1. Recursive top-down SED (> sed_tol_m) + arc-length stride (> arc_stride_m)
    const double tol_sq = cfg.sed_tol_m * cfg.sed_tol_m;
    std::vector<std::pair<std::size_t, std::size_t>> stack;
    stack.reserve(64);
    stack.emplace_back(0, n - 1);

    while (!stack.empty()) {
        const auto [a, b] = stack.back();
        stack.pop_back();
        if (b <= a + 1)
            continue;

        const double t0 = static_cast<double>(traj.t[a]);
        const double dt = static_cast<double>(traj.t[b]) - t0;
        const double inv_dt = dt > 0.0 ? 1.0 / dt : 0.0;
        const double x0 = traj.x[a], y0 = traj.y[a];
        const double seg_dx = traj.x[b] - x0, seg_dy = traj.y[b] - y0;

        double max_sq = -1.0;
        std::size_t best_k = a + 1;
        for (std::size_t k = a + 1; k < b; ++k) {
            const double c = (static_cast<double>(traj.t[k]) - t0) * inv_dt;
            const double dx = x0 + c * seg_dx - traj.x[k];
            const double dy = y0 + c * seg_dy - traj.y[k];
            const double d2 = dx * dx + dy * dy;
            if (d2 > max_sq) {
                max_sq = d2;
                best_k = k;
            }
        }

        if (max_sq > tol_sq) {
            keep[best_k] = 1;
            stack.emplace_back(a, best_k);
            stack.emplace_back(best_k, b);
        } else if (cfg.arc_stride_m > 0.0 && (arc[b] - arc[a]) > cfg.arc_stride_m) {
            const double mid_arc = 0.5 * (arc[a] + arc[b]);
            auto it = std::lower_bound(arc.begin() + static_cast<std::ptrdiff_t>(a + 1),
                                       arc.begin() + static_cast<std::ptrdiff_t>(b), mid_arc);
            std::size_t mid_k = static_cast<std::size_t>(it - arc.begin());
            mid_k = std::clamp(mid_k, a + 1, b - 1);
            keep[mid_k] = 1;
            stack.emplace_back(a, mid_k);
            stack.emplace_back(mid_k, b);
        }
    }

    // 2. Spatial cell-run midpoints (protects point-in-box Range coverage)
    if (cfg.cell_m > 0.0) {
        const double inv_cell = 1.0 / cfg.cell_m;
        auto cell_of = [&](std::size_t k) {
            return std::pair<std::int64_t, std::int64_t>{
                static_cast<std::int64_t>(std::floor(traj.x[k] * inv_cell)),
                static_cast<std::int64_t>(std::floor(traj.y[k] * inv_cell)),
            };
        };
        std::size_t run_start = 0;
        auto prev = cell_of(0);
        for (std::size_t k = 1; k < n; ++k) {
            const auto cur = cell_of(k);
            if (cur != prev) {
                keep[(run_start + k - 1) / 2] = 1;
                run_start = k;
                prev = cur;
            }
        }
        keep[(run_start + n - 1) / 2] = 1;
    }

    std::vector<std::size_t> pool;
    pool.reserve(n / 4 + 8);
    for (std::size_t k = 0; k < n; ++k)
        if (keep[k])
            pool.push_back(k);
    return pool;
}

} // namespace student
