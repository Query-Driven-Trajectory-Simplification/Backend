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

// Fast top-down PED or SED probe up to budget = 8: O(n log 8)
// Returns (max_err_at_b2, max_err_at_b8).
std::pair<double, double> probe_err_2_and_8(const core::TrajectoryView &tr, bool use_sed) {
    const std::size_t n = tr.size();
    if (n <= 2)
        return {0.0, 0.0};

    struct Seg {
        double max_sq;
        std::size_t argmax;
        std::size_t a;
        std::size_t b;
        bool operator<(const Seg &o) const { return max_sq < o.max_sq; }
    };

    auto eval_seg = [&](std::size_t a, std::size_t b) {
        Seg s{0.0, a + 1, a, b};
        if (use_sed) {
            const double t0 = static_cast<double>(tr.t[a]);
            const double dt = static_cast<double>(tr.t[b]) - t0;
            const double inv_dt = dt > 0.0 ? 1.0 / dt : 0.0;
            const double x0 = tr.x[a], y0 = tr.y[a];
            const double dx_seg = tr.x[b] - x0, dy_seg = tr.y[b] - y0;
            for (std::size_t k = a + 1; k < b; ++k) {
                const double c = (static_cast<double>(tr.t[k]) - t0) * inv_dt;
                const double dx = x0 + c * dx_seg - tr.x[k];
                const double dy = y0 + c * dy_seg - tr.y[k];
                const double d2 = dx * dx + dy * dy;
                if (d2 > s.max_sq) {
                    s.max_sq = d2;
                    s.argmax = k;
                }
            }
        } else {
            for (std::size_t k = a + 1; k < b; ++k) {
                const double d2 = selection::distance::squared_segment_distance(
                    tr.x[k], tr.y[k], tr.x[a], tr.y[a], tr.x[b], tr.y[b]);
                if (d2 > s.max_sq) {
                    s.max_sq = d2;
                    s.argmax = k;
                }
            }
        }
        return s;
    };

    std::priority_queue<Seg> heap;
    heap.push(eval_seg(0, n - 1));
    const double err2 = std::sqrt(heap.top().max_sq);

    const std::size_t lim = std::min<std::size_t>(n, 8);
    double err8 = err2;
    for (std::size_t b = 3; b <= lim; ++b) {
        if (heap.empty()) {
            err8 = 0.0;
            break;
        }
        const Seg top = heap.top();
        heap.pop();
        if (top.argmax - top.a >= 2)
            heap.push(eval_seg(top.a, top.argmax));
        if (top.b - top.argmax >= 2)
            heap.push(eval_seg(top.argmax, top.b));
        err8 = heap.empty() ? 0.0 : std::sqrt(heap.top().max_sq);
    }
    return {err2, n <= 8 ? 0.0 : err8};
}

struct Sig5 {
    std::array<double, 5> x{};
    std::array<double, 5> y{};
};

Sig5 compute_sig5(const core::TrajectoryView &tr, std::span<const double> arc) {
    Sig5 s;
    const std::size_t n = tr.size();
    const double total = arc.back();
    for (std::size_t m = 0; m < 5; ++m) {
        const double target = total * (static_cast<double>(m) * 0.25);
        auto it = std::lower_bound(arc.begin(), arc.end(), target);
        std::size_t idx = static_cast<std::size_t>(it - arc.begin());
        if (idx >= n)
            idx = n - 1;
        if (idx == 0 || arc[idx] == arc[idx - 1]) {
            s.x[m] = tr.x[idx];
            s.y[m] = tr.y[idx];
        } else {
            const double u = (target - arc[idx - 1]) / (arc[idx] - arc[idx - 1]);
            s.x[m] = tr.x[idx - 1] + u * (tr.x[idx] - tr.x[idx - 1]);
            s.y[m] = tr.y[idx - 1] + u * (tr.y[idx] - tr.y[idx - 1]);
        }
    }
    return s;
}

inline std::uint64_t pack_grid2d(std::int32_t gx, std::int32_t gy) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(gx)) << 32) |
           static_cast<std::uint64_t>(static_cast<std::uint32_t>(gy));
}

} // namespace

TrajectoryFeatureTable extract_trajectory_features(const core::TrajectoryDb &db,
                                                   const DbDensityIndex &density_idx,
                                                   const SampledRangeIndex *range_idx) {
    TrajectoryFeatureTable table;
    table.num_trajs = db.size();

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

    const std::size_t F = table.feature_names.size();
    table.values.assign(db.size() * F, 0.0f);

    // Precompute arc-lengths, 5-point signatures, and start-point spatial buckets
    std::vector<double> total_arcs(db.size(), 0.0);
    std::vector<Sig5> sigs(db.size());
    core::parallel_for(db.size(), [&](std::size_t i) {
        const auto tr = db.trajectory(i);
        const auto arc = compute_arc_length(tr);
        total_arcs[i] = arc.back();
        sigs[i] = compute_sig5(tr, arc);
    });
    std::unordered_map<std::uint64_t, std::vector<std::size_t>> fine_start_grid;   // 500m
    std::unordered_map<std::uint64_t, std::vector<std::size_t>> coarse_start_grid; // 2500m
    fine_start_grid.reserve(db.size());
    coarse_start_grid.reserve(db.size() / 2 + 16);
    for (std::size_t i = 0; i < db.size(); ++i) {
        const double sx = sigs[i].x[0], sy = sigs[i].y[0];
        fine_start_grid[pack_grid2d(static_cast<std::int32_t>(std::floor(sx / 500.0)),
                                    static_cast<std::int32_t>(std::floor(sy / 500.0)))]
            .push_back(i);
        coarse_start_grid[pack_grid2d(static_cast<std::int32_t>(std::floor(sx / 2500.0)),
                                      static_cast<std::int32_t>(std::floor(sy / 2500.0)))]
            .push_back(i);
    }

    const double log_db = std::log1p(static_cast<double>(std::max<std::size_t>(1, db.size())));

    core::parallel_for(db.size(), [&](std::size_t i) {
        const auto tr = db.trajectory(i);
        const std::size_t n = tr.size();
        float *row = table.values.data() + i * F;
        std::size_t col = 0;
        auto put = [&](double v) {
            row[col++] = std::isfinite(v) ? static_cast<float>(v) : 0.0f;
        };

        // Single-trip features (6)
        std::unordered_set<std::uint64_t> cells1k;
        cells1k.reserve(16);
        std::uint64_t prev_cell = 0;
        for (std::size_t k = 0; k < n; ++k) {
            const std::uint64_t key =
                pack_grid2d(static_cast<std::int32_t>(std::floor(tr.x[k] / 1000.0)),
                            static_cast<std::int32_t>(std::floor(tr.y[k] / 1000.0)));
            if (k == 0 || key != prev_cell) {
                cells1k.insert(key);
                prev_cell = key;
            }
        }

        const auto [rdp2, rdp8] = probe_err_2_and_8(tr, false);
        const auto [sed2, sed8] = probe_err_2_and_8(tr, true);
        (void)sed2;

        put(std::log(static_cast<double>(std::max<std::size_t>(1, n))));
        put(std::log1p(total_arcs[i]));
        put(std::log1p(static_cast<double>(cells1k.size())));
        put(std::log1p(rdp2));
        put(std::log1p(rdp8));
        put(std::log1p(sed8));

        // Database-context features (4, zero Frechet)
        const auto rho = density_idx.trajectory_range_rho(tr);
        double rho_sum = 0.0;
        for (float v : rho)
            rho_sum += v;
        const double rho_mean = rho.empty() ? 0.0 : rho_sum / static_cast<double>(rho.size());
        put(rho_mean);
        put(std::log1p(rho_mean * static_cast<double>(cells1k.size())));

        // 5-point signature neighbor count <= 200m and nearest neighbor distance d_1
        const auto &si = sigs[i];
        std::size_t cnt200 = 0;
        double d1 = std::numeric_limits<double>::infinity();
        auto scan_grid = [&](const auto &grid, double cell_m, bool count_near) {
            const auto gx = static_cast<std::int32_t>(std::floor(si.x[0] / cell_m));
            const auto gy = static_cast<std::int32_t>(std::floor(si.y[0] / cell_m));
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    const auto it = grid.find(pack_grid2d(gx + dx, gy + dy));
                    if (it == grid.end())
                        continue;
                    for (std::size_t j : it->second) {
                        if (j == i)
                            continue;
                        const auto &sj = sigs[j];
                        double d5 = 0.0;
                        for (std::size_t m = 0; m < 5; ++m)
                            d5 = std::max(d5, std::hypot(si.x[m] - sj.x[m], si.y[m] - sj.y[m]));
                        if (count_near && d5 <= 200.0)
                            ++cnt200;
                        if (d5 < d1)
                            d1 = d5;
                    }
                }
            }
        };
        scan_grid(fine_start_grid, 500.0, true);
        if (!std::isfinite(d1))
            scan_grid(coarse_start_grid, 2500.0, false);
        if (!std::isfinite(d1))
            d1 = 5000.0;

        put(std::log1p(static_cast<double>(cnt200)) / log_db);
        put(std::log1p(d1));

        // Exact sampled Range features (scale-invariant across DB size and box count)
        if (range_idx != nullptr && range_idx->valid_boxes > 0) {
            const double uw = range_idx->uncov_weight[i];
            const double ew = range_idx->ep_weight[i];
            put(std::log1p(uw));
            put(std::log1p(range_idx->uncov_count_norm[i]));
            put((uw + ew > 1e-12) ? (ew / (uw + ew)) : 1.0);
        } else {
            put(0.0);
            put(0.0);
            put(0.0);
        }
    });

    return table;
}

} // namespace student
