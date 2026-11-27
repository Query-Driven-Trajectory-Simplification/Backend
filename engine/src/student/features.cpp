#include "backend/engine/student/features.hpp"

#include "backend/engine/core/parallel.hpp"
#include "backend/engine/selection/distance_functions.hpp"
#include "backend/engine/student/candidate_pool.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace student {

namespace {

// Fast top-down PED or SED probe up to budget = 8: O(n log 8).
// Returns (max_err_at_budget_2, max_err_at_budget_8).
std::pair<double, double> probe_error_at_2_and_8(const core::TrajectoryView &traj, bool use_sed) {
    const std::size_t num_pts = traj.size();
    if (num_pts <= 2)
        return {0.0, 0.0};

    struct Segment {
        double max_err_sq;
        std::size_t furthest_idx;
        std::size_t start_idx;
        std::size_t end_idx;
        bool operator<(const Segment &other) const { return max_err_sq < other.max_err_sq; }
    };

    auto eval_segment = [&](std::size_t start_idx, std::size_t end_idx) {
        Segment seg{0.0, start_idx + 1, start_idx, end_idx};
        if (use_sed) {
            const double t_start = static_cast<double>(traj.t[start_idx]);
            const double dt = static_cast<double>(traj.t[end_idx]) - t_start;
            const double inv_dt = dt > 0.0 ? 1.0 / dt : 0.0;
            const double x_start = traj.x[start_idx], y_start = traj.y[start_idx];
            const double seg_dx = traj.x[end_idx] - x_start, seg_dy = traj.y[end_idx] - y_start;
            for (std::size_t pt_idx = start_idx + 1; pt_idx < end_idx; ++pt_idx) {
                const double frac = (static_cast<double>(traj.t[pt_idx]) - t_start) * inv_dt;
                const double dx = x_start + frac * seg_dx - traj.x[pt_idx];
                const double dy = y_start + frac * seg_dy - traj.y[pt_idx];
                const double err_sq = dx * dx + dy * dy;
                if (err_sq > seg.max_err_sq) {
                    seg.max_err_sq = err_sq;
                    seg.furthest_idx = pt_idx;
                }
            }
        } else {
            for (std::size_t pt_idx = start_idx + 1; pt_idx < end_idx; ++pt_idx) {
                const double err_sq = selection::distance::squared_segment_distance(
                    traj.x[pt_idx], traj.y[pt_idx], traj.x[start_idx], traj.y[start_idx],
                    traj.x[end_idx], traj.y[end_idx]);
                if (err_sq > seg.max_err_sq) {
                    seg.max_err_sq = err_sq;
                    seg.furthest_idx = pt_idx;
                }
            }
        }
        return seg;
    };

    std::priority_queue<Segment> heap;
    heap.push(eval_segment(0, num_pts - 1));
    const double err_at_2 = std::sqrt(heap.top().max_err_sq);

    const std::size_t max_budget = std::min<std::size_t>(num_pts, 8);
    double err_at_8 = err_at_2;
    for (std::size_t budget = 3; budget <= max_budget; ++budget) {
        if (heap.empty()) {
            err_at_8 = 0.0;
            break;
        }
        const Segment worst = heap.top();
        heap.pop();
        if (worst.furthest_idx - worst.start_idx >= 2)
            heap.push(eval_segment(worst.start_idx, worst.furthest_idx));
        if (worst.end_idx - worst.furthest_idx >= 2)
            heap.push(eval_segment(worst.furthest_idx, worst.end_idx));
        err_at_8 = heap.empty() ? 0.0 : std::sqrt(heap.top().max_err_sq);
    }
    return {err_at_2, num_pts <= 8 ? 0.0 : err_at_8};
}

struct ArcSignature5 {
    std::array<double, 5> x{};
    std::array<double, 5> y{};
};

ArcSignature5 compute_arc_signature5(const core::TrajectoryView &traj,
                                     std::span<const double> arc_length) {
    ArcSignature5 sig;
    const std::size_t num_pts = traj.size();
    const double total_arc = arc_length.back();
    for (std::size_t sample_idx = 0; sample_idx < 5; ++sample_idx) {
        const double target_arc = total_arc * (static_cast<double>(sample_idx) * 0.25);
        auto it = std::lower_bound(arc_length.begin(), arc_length.end(), target_arc);
        std::size_t pt_idx = static_cast<std::size_t>(it - arc_length.begin());
        if (pt_idx >= num_pts)
            pt_idx = num_pts - 1;
        if (pt_idx == 0 || arc_length[pt_idx] == arc_length[pt_idx - 1]) {
            sig.x[sample_idx] = traj.x[pt_idx];
            sig.y[sample_idx] = traj.y[pt_idx];
        } else {
            const double frac = (target_arc - arc_length[pt_idx - 1]) /
                                (arc_length[pt_idx] - arc_length[pt_idx - 1]);
            sig.x[sample_idx] = traj.x[pt_idx - 1] + frac * (traj.x[pt_idx] - traj.x[pt_idx - 1]);
            sig.y[sample_idx] = traj.y[pt_idx - 1] + frac * (traj.y[pt_idx] - traj.y[pt_idx - 1]);
        }
    }
    return sig;
}

inline std::uint64_t pack_grid2d(std::int32_t gx, std::int32_t gy) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(gx)) << 32) |
           static_cast<std::uint64_t>(static_cast<std::uint32_t>(gy));
}

} // namespace

TrajectoryFeatureTable extract_trajectory_features(const core::TrajectoryDb &db,
                                                   const DbDensityIndex &density_idx,
                                                   const SampledRangeIndex *range_idx) {
    const std::size_t num_trajs = db.size();
    TrajectoryFeatureTable table;
    table.num_trajs = num_trajs;

    // Compact 13-feature zero-Frechet set (6 single-trip + 7 database context)
    table.feature_names = {
        "log_n",
        "log_arc",
        "log_cells_1km",
        "log_rdp_err_2",
        "log_rdp_err_8",
        "log_sed_err_8",
        "rho_mean",
        "log_rho_sum",
        "sig5_cnt_200",
        "log_sig5_d1",
        "log_rng_uncov",
        "log_rng_boxes",
        "rng_ep_frac",
    };

    const std::size_t num_features = table.feature_names.size();
    table.values.assign(num_trajs * num_features, 0.0f);

    // Precompute arc-lengths, 5-point signatures, and start-point spatial buckets
    std::vector<double> total_arcs(num_trajs, 0.0);
    std::vector<ArcSignature5> signatures(num_trajs);
    core::parallel_for(num_trajs, [&](std::size_t traj_idx) {
        const auto traj = db.trajectory(traj_idx);
        const auto arc_length = compute_arc_length(traj);
        total_arcs[traj_idx] = arc_length.back();
        signatures[traj_idx] = compute_arc_signature5(traj, arc_length);
    });

    std::unordered_map<std::uint64_t, std::vector<std::size_t>> fine_start_grid;   // 500m
    std::unordered_map<std::uint64_t, std::vector<std::size_t>> coarse_start_grid; // 2500m
    fine_start_grid.reserve(num_trajs);
    coarse_start_grid.reserve(num_trajs / 2 + 16);
    for (std::size_t traj_idx = 0; traj_idx < num_trajs; ++traj_idx) {
        const double start_x = signatures[traj_idx].x[0];
        const double start_y = signatures[traj_idx].y[0];
        fine_start_grid[pack_grid2d(static_cast<std::int32_t>(std::floor(start_x / 500.0)),
                                    static_cast<std::int32_t>(std::floor(start_y / 500.0)))]
            .push_back(traj_idx);
        coarse_start_grid[pack_grid2d(static_cast<std::int32_t>(std::floor(start_x / 2500.0)),
                                      static_cast<std::int32_t>(std::floor(start_y / 2500.0)))]
            .push_back(traj_idx);
    }

    const double log_db_size = std::log1p(static_cast<double>(std::max<std::size_t>(1, num_trajs)));

    core::parallel_for(num_trajs, [&](std::size_t traj_idx) {
        const auto traj = db.trajectory(traj_idx);
        const std::size_t num_pts = traj.size();
        float *feature_row = table.values.data() + traj_idx * num_features;
        std::size_t col = 0;
        auto push_feature = [&](double val) {
            feature_row[col++] = std::isfinite(val) ? static_cast<float>(val) : 0.0f;
        };

        // Single-trip features (6)
        std::unordered_set<std::uint64_t> cells_1km;
        cells_1km.reserve(16);
        std::uint64_t prev_cell = 0;
        for (std::size_t pt_idx = 0; pt_idx < num_pts; ++pt_idx) {
            const std::uint64_t key =
                pack_grid2d(static_cast<std::int32_t>(std::floor(traj.x[pt_idx] / 1000.0)),
                            static_cast<std::int32_t>(std::floor(traj.y[pt_idx] / 1000.0)));
            if (pt_idx == 0 || key != prev_cell) {
                cells_1km.insert(key);
                prev_cell = key;
            }
        }

        const auto [rdp_err_2, rdp_err_8] = probe_error_at_2_and_8(traj, false);
        const auto [sed_err_2, sed_err_8] = probe_error_at_2_and_8(traj, true);
        (void)sed_err_2;

        push_feature(std::log(static_cast<double>(std::max<std::size_t>(1, num_pts))));
        push_feature(std::log1p(total_arcs[traj_idx]));
        push_feature(std::log1p(static_cast<double>(cells_1km.size())));
        push_feature(std::log1p(rdp_err_2));
        push_feature(std::log1p(rdp_err_8));
        push_feature(std::log1p(sed_err_8));

        // Database-context features (7, zero Frechet)
        const auto rho = density_idx.trajectory_range_rho(traj);
        double rho_sum = 0.0;
        for (float val : rho)
            rho_sum += val;
        const double rho_mean = rho.empty() ? 0.0 : rho_sum / static_cast<double>(rho.size());
        push_feature(rho_mean);
        push_feature(std::log1p(rho_mean * static_cast<double>(cells_1km.size())));

        // 5-point signature neighbor count <= 200m and nearest neighbor distance
        const auto &sig_i = signatures[traj_idx];
        std::size_t near_neighbor_count = 0;
        double nearest_sig_dist = std::numeric_limits<double>::infinity();
        auto scan_grid = [&](const auto &grid, double cell_m, bool count_near) {
            const auto gx = static_cast<std::int32_t>(std::floor(sig_i.x[0] / cell_m));
            const auto gy = static_cast<std::int32_t>(std::floor(sig_i.y[0] / cell_m));
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    const auto it = grid.find(pack_grid2d(gx + dx, gy + dy));
                    if (it == grid.end())
                        continue;
                    for (std::size_t other_idx : it->second) {
                        if (other_idx == traj_idx)
                            continue;
                        const auto &sig_j = signatures[other_idx];
                        double sig_dist = 0.0;
                        for (std::size_t m = 0; m < 5; ++m)
                            sig_dist = std::max(
                                sig_dist, std::hypot(sig_i.x[m] - sig_j.x[m], sig_i.y[m] - sig_j.y[m]));
                        if (count_near && sig_dist <= 200.0)
                            ++near_neighbor_count;
                        if (sig_dist < nearest_sig_dist)
                            nearest_sig_dist = sig_dist;
                    }
                }
            }
        };
        scan_grid(fine_start_grid, 500.0, true);
        if (!std::isfinite(nearest_sig_dist))
            scan_grid(coarse_start_grid, 2500.0, false);
        if (!std::isfinite(nearest_sig_dist))
            nearest_sig_dist = 5000.0;

        push_feature(std::log1p(static_cast<double>(near_neighbor_count)) / log_db_size);
        push_feature(std::log1p(nearest_sig_dist));

        // Exact sampled Range features (scale-invariant across DB size and box count)
        if (range_idx != nullptr && range_idx->valid_boxes > 0) {
            const double uncovered_weight = range_idx->uncov_weight[traj_idx];
            const double endpoint_weight = range_idx->ep_weight[traj_idx];
            const double total_weight = uncovered_weight + endpoint_weight;
            push_feature(std::log1p(uncovered_weight));
            push_feature(std::log1p(range_idx->uncov_count_norm[traj_idx]));
            push_feature((total_weight > 1e-12) ? (endpoint_weight / total_weight) : 1.0);
        } else {
            push_feature(0.0);
            push_feature(0.0);
            push_feature(0.0);
        }
    });

    return table;
}

} // namespace student
