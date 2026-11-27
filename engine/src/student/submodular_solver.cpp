#include "backend/engine/student/submodular_solver.hpp"

#include "backend/engine/selection/distance_functions.hpp"
#include "backend/engine/student/candidate_pool.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <unordered_map>

namespace student {

namespace {

// Exact 25m -> 200m linear ramp in [0, 1] matching Where-At & Shape
inline double ramp_unit(double d) {
    if (d <= 25.0)
        return 1.0;
    if (d >= 200.0)
        return 0.0;
    return (200.0 - d) / 175.0;
}

struct Probe {
    double t;
    double x;
    double y;
};

std::vector<Probe> build_probes(const core::TrajectoryView &tr, std::size_t count = 64) {
    const std::size_t n = tr.size();
    if (n < 2)
        return {};
    const double t0 = static_cast<double>(tr.t.front());
    const double t1 = static_cast<double>(tr.t.back());
    if (t1 <= t0)
        return {};

    // Collect eligible intervals (gap <= 60s, matching geolife.json where_at.max_gap_seconds)
    std::vector<double> cum_dur;
    std::vector<std::size_t> seg_idx;
    cum_dur.reserve(n);
    seg_idx.reserve(n);
    double total = 0.0;
    for (std::size_t k = 0; k + 1 < n; ++k) {
        const double dt = static_cast<double>(tr.t[k + 1] - tr.t[k]);
        if (dt > 0.0 && dt <= 60.0) {
            total += dt;
            cum_dur.push_back(total);
            seg_idx.push_back(k);
        }
    }
    std::vector<Probe> probes;
    probes.reserve(count);
    if (total <= 0.0) {
        // Fallback uniform over full time span
        for (std::size_t p = 0; p < count; ++p) {
            const double tp = t0 + ((static_cast<double>(p) + 0.5) / static_cast<double>(count)) * (t1 - t0);
            auto it = std::lower_bound(tr.t.begin(), tr.t.end(), static_cast<std::int64_t>(tp));
            std::size_t j = static_cast<std::size_t>(it - tr.t.begin());
            if (j == 0)
                j = 1;
            if (j >= n)
                j = n - 1;
            const double dt = static_cast<double>(tr.t[j] - tr.t[j - 1]);
            const double u = dt > 0.0 ? std::clamp((tp - static_cast<double>(tr.t[j - 1])) / dt, 0.0, 1.0) : 0.0;
            probes.push_back({tp, tr.x[j - 1] + u * (tr.x[j] - tr.x[j - 1]),
                              tr.y[j - 1] + u * (tr.y[j] - tr.y[j - 1])});
        }
        return probes;
    }

    for (std::size_t p = 0; p < count; ++p) {
        const double target = ((static_cast<double>(p) + 0.5) / static_cast<double>(count)) * total;
        auto it = std::lower_bound(cum_dur.begin(), cum_dur.end(), target);
        std::size_t pos = static_cast<std::size_t>(it - cum_dur.begin());
        if (pos >= seg_idx.size())
            pos = seg_idx.size() - 1;
        const std::size_t k = seg_idx[pos];
        const double prev_cum = pos > 0 ? cum_dur[pos - 1] : 0.0;
        const double dt = static_cast<double>(tr.t[k + 1] - tr.t[k]);
        const double u = dt > 0.0 ? std::clamp((target - prev_cum) / dt, 0.0, 1.0) : 0.0;
        const double tp = static_cast<double>(tr.t[k]) + u * dt;
        probes.push_back({tp, tr.x[k] + u * (tr.x[k + 1] - tr.x[k]),
                          tr.y[k] + u * (tr.y[k + 1] - tr.y[k])});
    }
    return probes;
}

struct CellModel {
    // Each unique 333m x 3.5d cell within +-500m x +-3.5d of the pool gets a compact ID 0..C-1
    std::vector<double> cell_weight;
    double total_weight = 1.0;
    // For each candidate j in 0..M-1: list of unique cell IDs covered by placing a knot at pool[j]
    // (a knot at p covers any visited cell whose center is within 500m in x & y and 3.5d in t,
    //  approximating the 1km x 1km x 7d point-in-box footprint!)
    std::vector<std::vector<std::uint16_t>> cand_covers;
};

CellModel build_cell_model(const core::TrajectoryView &tr, std::span<const std::size_t> pool,
                           const DbDensityIndex &density_idx) {
    CellModel cm;
    const std::size_t m = pool.size();
    cm.cand_covers.resize(m);
    if (m == 0)
        return cm;

    constexpr double kHalfBox = 500.0;     // 1km box half-width
    constexpr double kHalfTime = 302400.0; // 7-day window half-width
    const double cell_m = density_idx.cell_m();
    const double inv_cell = 1.0 / cell_m;
    const double time_bin = static_cast<double>(density_idx.time_bin_s());
    const double inv_time = 1.0 / time_bin;

    std::unordered_map<std::uint64_t, std::uint16_t> key_to_id;
    cm.cell_weight.reserve(128);
    double wsum = 0.0;

    for (std::size_t j = 0; j < m; ++j) {
        const std::size_t k = pool[j];
        const double px = tr.x[k], py = tr.y[k], pt = static_cast<double>(tr.t[k]);
        // Exact integer cell bounds satisfying |p - (c + 0.5) * step| <= kHalf
        const auto cx_lo = static_cast<std::int32_t>(std::ceil((px - kHalfBox) * inv_cell - 0.5));
        const auto cx_hi = static_cast<std::int32_t>(std::floor((px + kHalfBox) * inv_cell - 0.5));
        const auto cy_lo = static_cast<std::int32_t>(std::ceil((py - kHalfBox) * inv_cell - 0.5));
        const auto cy_hi = static_cast<std::int32_t>(std::floor((py + kHalfBox) * inv_cell - 0.5));
        const auto ct_lo = static_cast<std::int32_t>(std::ceil((pt - kHalfTime) * inv_time - 0.5));
        const auto ct_hi = static_cast<std::int32_t>(std::floor((pt + kHalfTime) * inv_time - 0.5));

        auto &cov = cm.cand_covers[j];
        for (std::int32_t cx_i = cx_lo; cx_i <= cx_hi; ++cx_i) {
            for (std::int32_t cy_i = cy_lo; cy_i <= cy_hi; ++cy_i) {
                for (std::int32_t ct_i = ct_lo; ct_i <= ct_hi; ++ct_i) {
                    const double cx = (static_cast<double>(cx_i) + 0.5) * cell_m;
                    const double cy = (static_cast<double>(cy_i) + 0.5) * cell_m;
                    const double ct = (static_cast<double>(ct_i) + 0.5) * time_bin;
                    if (std::abs(px - cx) > kHalfBox || std::abs(py - cy) > kHalfBox ||
                        std::abs(pt - ct) > kHalfTime)
                        continue;

                    const std::uint64_t key = DbDensityIndex::pack_3d(cx_i, cy_i, ct_i);
                    auto it = key_to_id.find(key);
                    std::uint16_t id = 0;
                    if (it == key_to_id.end()) {
                        const auto st = density_idx.cell_stats(cx_i, cy_i, ct_i);
                        if (st.point_count == 0 || cm.cell_weight.size() >= 2500)
                            continue;
                        double tr_nbhd = static_cast<double>(st.traj_count);
                        for (int nx = -1; nx <= 1; ++nx)
                            for (int ny = -1; ny <= 1; ++ny)
                                if (nx != 0 || ny != 0)
                                    tr_nbhd += 0.35 * static_cast<double>(
                                                          density_idx.cell_stats(cx_i + nx, cy_i + ny, ct_i).traj_count);
                        const double acc = tr_nbhd < 3.0 ? 0.05 : std::min(1.0, tr_nbhd / 20.0);
                        const double w = static_cast<double>(st.point_count) * acc / std::max(3.0, tr_nbhd);
                        id = static_cast<std::uint16_t>(cm.cell_weight.size());
                        key_to_id.emplace(key, id);
                        cm.cell_weight.push_back(w);
                        wsum += w;
                    } else {
                        id = it->second;
                    }
                    cov.push_back(id);
                }
            }
        }
    }

    cm.total_weight = std::max(1e-6, wsum);
    return cm;
}

CellModel build_blended_range_model(const core::TrajectoryView &tr, std::span<const std::size_t> pool,
                                    const DbDensityIndex &density_idx,
                                    std::span<const SampledRangeBox> boxes, double uncov_frac) {
    CellModel cm;
    const std::size_t m = pool.size();
    cm.cand_covers.resize(m);
    if (m == 0 || boxes.empty())
        return cm; // Zero uncovered valid [5, 20] Range boxes -> spend budget on Geo & Where-At!

    // 1. 20% continuous 333m x 3.5d cell grid floor (never runs dry at 2% budget)
    cm = build_cell_model(tr, pool, density_idx);
    const double cell_scale = 0.20 / cm.total_weight;
    for (double &w : cm.cell_weight)
        w *= cell_scale;

    // 2. 80% exact sampled 1km x 7d uncovered Range boxes scaled by true uncovered fraction
    //    (so a trip already 90% covered by endpoints doesn't steal points from Where-At / Join!)
    const std::size_t base_id = cm.cell_weight.size();
    const std::size_t B = std::min<std::size_t>(boxes.size(), 65000 - base_id);
    double box_wsum = 0.0;
    for (std::size_t b = 0; b < B; ++b) {
        const auto &sb = boxes[b];
        const auto &bx = sb.box;
        bool any = false;
        const auto id = static_cast<std::uint16_t>(cm.cell_weight.size());
        for (std::size_t j = 0; j < m; ++j) {
            const std::size_t k = pool[j];
            const std::int64_t tk = tr.t[k];
            if (tk < bx.t_min)
                continue;
            if (tk > bx.t_max)
                break;
            const double xk = tr.x[k], yk = tr.y[k];
            if (xk >= bx.x_min && xk <= bx.x_max && yk >= bx.y_min && yk <= bx.y_max) {
                cm.cand_covers[j].push_back(id);
                any = true;
            }
        }
        if (any) {
            cm.cell_weight.push_back(sb.weight);
            box_wsum += sb.weight;
        }
    }
    if (box_wsum > 1e-9) {
        const double box_scale = (0.80 * std::clamp(uncov_frac, 0.0, 1.0)) / box_wsum;
        for (std::size_t idx = base_id; idx < cm.cell_weight.size(); ++idx)
            cm.cell_weight[idx] *= box_scale;
    }
    cm.total_weight = 1.0;
    return cm;
}

struct SegScores {
    double shape_arc_sum = 0.0; // exact trapezoidal arc-weighted shape in [0, seg_arc]
    double ped_l1_sum = 0.0;    // exact trapezoidal arc-weighted L1 perpendicular error (m * m)
    double max_dev = 0.0;       // exact max perpendicular segment distance (m)
    double wa_sum = 0.0;        // sum of ramp_unit over probes in [t_a, t_b)
};

SegScores eval_seg_scores(const core::TrajectoryView &tr, std::span<const std::size_t> pool,
                          std::span<const double> arc_pool, std::span<const Probe> probes,
                          std::span<const std::size_t> probe_start_of_cand, std::size_t ia,
                          std::size_t ib) {
    SegScores sc;
    const std::size_t a = pool[ia], b = pool[ib];
    const double ax = tr.x[a], ay = tr.y[a], bx = tr.x[b], by = tr.y[b];
    const double seg_dx = bx - ax, seg_dy = by - ay;

    // 1. Exact perpendicular segment distance + trapezoidal arc-weighted shape & L1 error
    double mx = 0.0;
    double arc_sum = 0.0;
    double l1_sum = 0.0;
    double prev_r = 1.0;
    double prev_d = 0.0;
    for (std::size_t ik = ia + 1; ik <= ib; ++ik) {
        const double w = arc_pool[ik] - arc_pool[ik - 1];
        double r = 1.0;
        double d = 0.0;
        if (ik < ib) {
            const std::size_t k = pool[ik];
            d = std::sqrt(selection::distance::squared_segment_distance(tr.x[k], tr.y[k], ax, ay, bx, by));
            if (d > mx)
                mx = d;
            r = ramp_unit(d);
        }
        if (w > 0.0) {
            arc_sum += w * 0.5 * (prev_r + r);
            l1_sum += w * 0.5 * (prev_d + d);
        }
        prev_r = r;
        prev_d = d;
    }
    sc.shape_arc_sum = arc_sum;
    sc.ped_l1_sum = l1_sum;
    sc.max_dev = mx;

    // 2. Where-At over probes in [probe_start_of_cand[ia], probe_start_of_cand[ib])
    const std::size_t p0 = probe_start_of_cand[ia];
    const std::size_t p1 = probe_start_of_cand[ib];
    const double t0 = static_cast<double>(tr.t[a]);
    const double dt = static_cast<double>(tr.t[b] - tr.t[a]);
    const double inv_dt = dt > 0.0 ? 1.0 / dt : 0.0;
    double wa = 0.0;
    for (std::size_t p = p0; p < p1; ++p) {
        const double u = std::clamp((probes[p].t - t0) * inv_dt, 0.0, 1.0);
        const double ix = ax + u * seg_dx;
        const double iy = ay + u * seg_dy;
        const double d = std::hypot(ix - probes[p].x, iy - probes[p].y);
        wa += ramp_unit(d);
    }
    sc.wa_sum = wa;
    return sc;
}

} // namespace

std::vector<std::size_t>
submodular_simplify_on_pool(const core::TrajectoryView &traj, std::span<const std::size_t> pool,
                            std::size_t budget, const SubmodularWeights &w,
                            const DbDensityIndex &density_idx,
                            const std::vector<SampledRangeBox> *uncov_boxes, double uncov_frac) {
    const std::size_t n = traj.size();
    const std::size_t m = pool.size();
    if (budget >= n) {
        std::vector<std::size_t> all(n);
        for (std::size_t k = 0; k < n; ++k)
            all[k] = k;
        return all;
    }
    budget = std::clamp(budget, std::size_t{2}, n);
    if (budget == 2 || m <= 2)
        return {0, n - 1};
    if (budget >= m) {
        std::vector<std::size_t> out(pool.begin(), pool.end());
        if (out.size() < budget) {
            std::vector<std::uint8_t> used(n, 0);
            for (std::size_t idx : out)
                used[idx] = 1;
            for (std::size_t k = 1; k + 1 < n && out.size() < budget; ++k) {
                if (!used[k]) {
                    used[k] = 1;
                    out.push_back(k);
                }
            }
            std::ranges::sort(out);
        }
        return out;
    }

    const auto full_arc = compute_arc_length(traj);
    std::vector<double> arc_pool(m);
    for (std::size_t j = 0; j < m; ++j)
        arc_pool[j] = full_arc[pool[j]];
    const double inv_total_arc = 1.0 / std::max(1.0, arc_pool.back());

    const auto probes = build_probes(traj, 64);
    const double inv_probes = probes.empty() ? 0.0 : 1.0 / static_cast<double>(probes.size());
    std::vector<std::size_t> probe_start(m, probes.size());
    {
        std::size_t p = 0;
        for (std::size_t j = 0; j < m; ++j) {
            const double tj = static_cast<double>(traj.t[pool[j]]);
            while (p < probes.size() && probes[p].t < tj)
                ++p;
            probe_start[j] = p;
        }
        probe_start[m - 1] = probes.size();
    }

    const auto cm = (uncov_boxes != nullptr)
                        ? build_blended_range_model(traj, pool, density_idx, *uncov_boxes, uncov_frac)
                        : build_cell_model(traj, pool, density_idx);
    const std::size_t C = cm.cell_weight.size();
    std::vector<std::uint16_t> cov_count(C, 0);

    auto add_cover = [&](std::size_t j) {
        for (std::uint16_t c : cm.cand_covers[j])
            cov_count[c] += 1;
    };
    auto rem_cover = [&](std::size_t j) {
        for (std::uint16_t c : cm.cand_covers[j])
            cov_count[c] -= 1;
    };
    auto rng_gain_if_add = [&](std::size_t j) -> double {
        double g = 0.0;
        for (std::uint16_t c : cm.cand_covers[j])
            if (cov_count[c] == 0)
                g += cm.cell_weight[c];
        return g / cm.total_weight;
    };

    // Active kept pool indices (kept sorted)
    std::vector<std::size_t> kp;
    kp.reserve(budget);
    kp.push_back(0);
    kp.push_back(m - 1);
    add_cover(0);
    add_cover(m - 1);

    // Track current segment scores for each interval (kp[s], kp[s+1])
    std::vector<SegScores> segs;
    segs.reserve(budget);
    segs.push_back(eval_seg_scores(traj, pool, arc_pool, probes, probe_start, 0, m - 1));

    // Greedy submodular insertion: at each step pick the candidate maximizing
    //   w_geo * delta_shape + w_wa * delta_wa + w_rng * delta_rng_cells
    // To keep runtime O(budget * sqrt(m)), scan up to 16 evenly spaced + local peak candidates
    // inside each active segment (always including the max-PED, max-SED, and highest-uncovered-rho
    // points of the segment!).
    while (kp.size() < budget) {
        double best_gain = -1e18;
        std::size_t best_seg = 0;
        std::size_t best_ik = 0;

        // Current whole-trajectory max arc-Frechet deviation (for L_inf Shape term)
        double global_max_dev = 0.0;
        std::size_t max_dev_seg = 0;
        for (std::size_t s = 0; s < segs.size(); ++s) {
            if (segs[s].max_dev > global_max_dev) {
                global_max_dev = segs[s].max_dev;
                max_dev_seg = s;
            }
        }
        double second_max_dev = 0.0;
        for (std::size_t s = 0; s < segs.size(); ++s)
            if (s != max_dev_seg && segs[s].max_dev > second_max_dev)
                second_max_dev = segs[s].max_dev;

        for (std::size_t s = 0; s < segs.size(); ++s) {
            const std::size_t ia = kp[s], ib = kp[s + 1];
            if (ib <= ia + 1)
                continue;
            const auto &base = segs[s];
            const bool is_worst_seg = (s == max_dev_seg);

            // Pick a compact candidate set inside (ia, ib):
            // (a) max-arc-dev point, (b) max-SED point, (c) highest uncovered-range point, (d) uniform points
            std::size_t a = pool[ia], b = pool[ib];
            const double ax = traj.x[a], ay = traj.y[a], bx = traj.x[b], by = traj.y[b];
            const double t0 = static_cast<double>(traj.t[a]);
            const double dt = static_cast<double>(traj.t[b] - traj.t[a]);
            const double inv_dt = dt > 0.0 ? 1.0 / dt : 0.0;
            std::size_t arg_dev = ia + 1, arg_sed = ia + 1, arg_rng = ia + 1;
            double max_dev_sq = -1.0, max_sed_sq = -1.0, max_rng = -1.0;

            const std::size_t span = ib - ia - 1;
            for (std::size_t ik = ia + 1; ik < ib; ++ik) {
                const std::size_t k = pool[ik];
                const double dev2 =
                    selection::distance::squared_segment_distance(traj.x[k], traj.y[k], ax, ay, bx, by);
                if (dev2 > max_dev_sq) {
                    max_dev_sq = dev2;
                    arg_dev = ik;
                }
                const double u = (static_cast<double>(traj.t[k]) - t0) * inv_dt;
                const double sdx = ax + u * (bx - ax) - traj.x[k];
                const double sdy = ay + u * (by - ay) - traj.y[k];
                const double sed2 = sdx * sdx + sdy * sdy;
                if (sed2 > max_sed_sq) {
                    max_sed_sq = sed2;
                    arg_sed = ik;
                }
                const double rg = rng_gain_if_add(ik);
                if (rg > max_rng) {
                    max_rng = rg;
                    arg_rng = ik;
                }
            }

            std::array<std::size_t, 8> cands{};
            std::size_t n_cands = 0;
            cands[n_cands++] = arg_dev;
            cands[n_cands++] = arg_sed;
            cands[n_cands++] = arg_rng;
            const std::size_t n_uni = std::min<std::size_t>(5, span);
            for (std::size_t q = 1; q <= n_uni; ++q)
                cands[n_cands++] = ia + (q * (ib - ia)) / (n_uni + 1);
            std::sort(cands.begin(), cands.begin() + static_cast<std::ptrdiff_t>(n_cands));
            n_cands = static_cast<std::size_t>(
                std::unique(cands.begin(), cands.begin() + static_cast<std::ptrdiff_t>(n_cands)) -
                cands.begin());

            for (std::size_t ci = 0; ci < n_cands; ++ci) {
                const std::size_t ik = cands[ci];
                if (ik <= ia || ik >= ib)
                    continue;
                const auto left_sc =
                    eval_seg_scores(traj, pool, arc_pool, probes, probe_start, ia, ik);
                const auto right_sc =
                    eval_seg_scores(traj, pool, arc_pool, probes, probe_start, ik, ib);

                // Exact RDP point deviation (never vanishes on S-curves!) + L1 area drop + arc shape
                const std::size_t k = pool[ik];
                const double d_pt = std::sqrt(selection::distance::squared_segment_distance(
                    traj.x[k], traj.y[k], ax, ay, bx, by));
                const double d_l1_drop =
                    std::max(0.0, base.ped_l1_sum - (left_sc.ped_l1_sum + right_sc.ped_l1_sum)) *
                    inv_total_arc;
                const double d_arc =
                    (left_sc.shape_arc_sum + right_sc.shape_arc_sum - base.shape_arc_sum) *
                    inv_total_arc;
                const double new_seg_max = std::max(left_sc.max_dev, right_sc.max_dev);
                double d_global_drop = 0.0;
                if (is_worst_seg) {
                    const double new_global_max = std::max(second_max_dev, new_seg_max);
                    d_global_drop = std::max(0.0, global_max_dev - new_global_max) / 150.0;
                }
                const double d_geo =
                    0.50 * (d_pt / 150.0) + 0.30 * (d_l1_drop / 35.0) + 0.20 * d_global_drop + 0.40 * d_arc;

                const double d_wa =
                    (left_sc.wa_sum + right_sc.wa_sum - base.wa_sum) * inv_probes;
                const double d_rng = rng_gain_if_add(ik);

                const double gain = w.w_geo * d_geo + w.w_wa * d_wa + w.w_rng * d_rng;
                if (gain > best_gain) {
                    best_gain = gain;
                    best_seg = s;
                    best_ik = ik;
                }
            }
        }

        if (best_ik == 0)
            break;

        const std::size_t ia = kp[best_seg], ib = kp[best_seg + 1];
        kp.insert(kp.begin() + static_cast<std::ptrdiff_t>(best_seg + 1), best_ik);
        add_cover(best_ik);
        segs[best_seg] = eval_seg_scores(traj, pool, arc_pool, probes, probe_start, ia, best_ik);
        segs.insert(segs.begin() + static_cast<std::ptrdiff_t>(best_seg + 1),
                    eval_seg_scores(traj, pool, arc_pool, probes, probe_start, best_ik, ib));
    }

    // Fast Red-Black Free-Knot Polish optimizing the EXACT same 3-term objective!
    for (int pass = 0; pass < w.rb_passes; ++pass) {
        bool changed = false;
        for (int parity = 1; parity <= 2; ++parity) {
            for (std::size_t j = static_cast<std::size_t>(parity); j + 1 < kp.size(); j += 2) {
                const std::size_t il = kp[j - 1];
                const std::size_t ir = kp[j + 1];
                if (ir <= il + 2)
                    continue;
                const std::size_t orig_c = kp[j];

                // Compute max_dev across all OTHER segments for exact L_inf
                double other_max_dev = 0.0;
                for (std::size_t s = 0; s < segs.size(); ++s)
                    if (s != j - 1 && s != j && segs[s].max_dev > other_max_dev)
                        other_max_dev = segs[s].max_dev;

                rem_cover(orig_c);

                auto eval_c = [&](std::size_t c) -> double {
                    const auto sl = eval_seg_scores(traj, pool, arc_pool, probes, probe_start, il, c);
                    const auto sr = eval_seg_scores(traj, pool, arc_pool, probes, probe_start, c, ir);
                    const double geo_arc = (sl.shape_arc_sum + sr.shape_arc_sum) * inv_total_arc;
                    const double geo_l1 = (sl.ped_l1_sum + sr.ped_l1_sum) * inv_total_arc;
                    const double new_seg_max = std::max(sl.max_dev, sr.max_dev);
                    const double new_glob_max = std::max(other_max_dev, new_seg_max);
                    const double geo = 0.40 * geo_arc - 0.35 * (geo_l1 / 35.0) -
                                       0.35 * (new_seg_max / 150.0) - 0.25 * (new_glob_max / 150.0);
                    const double wa = (sl.wa_sum + sr.wa_sum) * inv_probes;
                    const double rng = rng_gain_if_add(c);
                    return w.w_geo * geo + w.w_wa * wa + w.w_rng * rng;
                };

                std::size_t best_c = orig_c;
                double best_val = eval_c(orig_c);

                // Coarse scan across (il, ir) + local halving
                const std::size_t span = ir - il - 1;
                const std::size_t n_coarse = std::min<std::size_t>(10, span);
                for (std::size_t q = 1; q <= n_coarse; ++q) {
                    const std::size_t c = il + (q * (ir - il)) / (n_coarse + 1);
                    if (c <= il || c >= ir || c == best_c)
                        continue;
                    const double v = eval_c(c);
                    if (v > best_val + 1e-9) {
                        best_val = v;
                        best_c = c;
                    }
                }
                std::size_t step = std::max<std::size_t>(1, span / (n_coarse + 1));
                while (step >= 1) {
                    bool imp = false;
                    if (best_c > il + step) {
                        const double v = eval_c(best_c - step);
                        if (v > best_val + 1e-9) {
                            best_val = v;
                            best_c -= step;
                            imp = true;
                        }
                    }
                    if (best_c + step < ir) {
                        const double v = eval_c(best_c + step);
                        if (v > best_val + 1e-9) {
                            best_val = v;
                            best_c += step;
                            imp = true;
                        }
                    }
                    if (!imp)
                        step /= 2;
                }

                kp[j] = best_c;
                add_cover(best_c);
                if (best_c != orig_c) {
                    segs[j - 1] = eval_seg_scores(traj, pool, arc_pool, probes, probe_start, il, best_c);
                    segs[j] = eval_seg_scores(traj, pool, arc_pool, probes, probe_start, best_c, ir);
                    changed = true;
                }
            }
        }
        if (!changed)
            break;
    }

    std::vector<std::size_t> out(kp.size());
    for (std::size_t j = 0; j < kp.size(); ++j)
        out[j] = pool[kp[j]];
    return out;
}

} // namespace student
