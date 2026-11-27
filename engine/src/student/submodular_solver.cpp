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
inline double ramp_unit(double dist_m) {
    if (dist_m <= 25.0)
        return 1.0;
    if (dist_m >= 200.0)
        return 0.0;
    return (200.0 - dist_m) / 175.0;
}

struct Probe {
    double t;
    double x;
    double y;
};

std::vector<Probe> build_probes(const core::TrajectoryView &traj, std::size_t count = 64) {
    const std::size_t traj_len = traj.size();
    if (traj_len < 2)
        return {};
    const double start_t = static_cast<double>(traj.t.front());
    const double end_t = static_cast<double>(traj.t.back());
    if (end_t <= start_t)
        return {};

    // Collect eligible intervals (gap <= 60s, matching geolife.json where_at.max_gap_seconds)
    std::vector<double> cumulative_duration;
    std::vector<std::size_t> segment_indices;
    cumulative_duration.reserve(traj_len);
    segment_indices.reserve(traj_len);
    double total_duration = 0.0;
    for (std::size_t idx = 0; idx + 1 < traj_len; ++idx) {
        const double dt = static_cast<double>(traj.t[idx + 1] - traj.t[idx]);
        if (dt > 0.0 && dt <= 60.0) {
            total_duration += dt;
            cumulative_duration.push_back(total_duration);
            segment_indices.push_back(idx);
        }
    }
    std::vector<Probe> probes;
    probes.reserve(count);
    if (total_duration <= 0.0) {
        // Fallback uniform over full time span
        for (std::size_t probe_idx = 0; probe_idx < count; ++probe_idx) {
            const double probe_t =
                start_t + ((static_cast<double>(probe_idx) + 0.5) / static_cast<double>(count)) *
                              (end_t - start_t);
            auto it = std::lower_bound(traj.t.begin(), traj.t.end(),
                                       static_cast<std::int64_t>(probe_t));
            std::size_t pt_idx = static_cast<std::size_t>(it - traj.t.begin());
            if (pt_idx == 0)
                pt_idx = 1;
            if (pt_idx >= traj_len)
                pt_idx = traj_len - 1;
            const double dt = static_cast<double>(traj.t[pt_idx] - traj.t[pt_idx - 1]);
            const double frac =
                dt > 0.0 ? std::clamp((probe_t - static_cast<double>(traj.t[pt_idx - 1])) / dt,
                                      0.0, 1.0)
                         : 0.0;
            probes.push_back({probe_t,
                              traj.x[pt_idx - 1] + frac * (traj.x[pt_idx] - traj.x[pt_idx - 1]),
                              traj.y[pt_idx - 1] + frac * (traj.y[pt_idx] - traj.y[pt_idx - 1])});
        }
        return probes;
    }

    for (std::size_t probe_idx = 0; probe_idx < count; ++probe_idx) {
        const double target_duration =
            ((static_cast<double>(probe_idx) + 0.5) / static_cast<double>(count)) * total_duration;
        auto it =
            std::lower_bound(cumulative_duration.begin(), cumulative_duration.end(), target_duration);
        std::size_t seg_pos = static_cast<std::size_t>(it - cumulative_duration.begin());
        if (seg_pos >= segment_indices.size())
            seg_pos = segment_indices.size() - 1;
        const std::size_t pt_idx = segment_indices[seg_pos];
        const double prev_duration = seg_pos > 0 ? cumulative_duration[seg_pos - 1] : 0.0;
        const double dt = static_cast<double>(traj.t[pt_idx + 1] - traj.t[pt_idx]);
        const double frac =
            dt > 0.0 ? std::clamp((target_duration - prev_duration) / dt, 0.0, 1.0) : 0.0;
        const double probe_t = static_cast<double>(traj.t[pt_idx]) + frac * dt;
        probes.push_back({probe_t,
                          traj.x[pt_idx] + frac * (traj.x[pt_idx + 1] - traj.x[pt_idx]),
                          traj.y[pt_idx] + frac * (traj.y[pt_idx + 1] - traj.y[pt_idx])});
    }
    return probes;
}

struct RangeCoverageModel {
    // Each unique coverage unit (333m cell or sampled 1km x 7d box) gets a compact ID 0..C-1
    std::vector<double> unit_weights;
    double total_weight = 1.0;
    // For each candidate j in 0..pool_size-1: list of unit IDs covered by placing a knot at pool[j]
    std::vector<std::vector<std::uint16_t>> candidate_covers;
};

RangeCoverageModel build_cell_model(const core::TrajectoryView &traj,
                                    std::span<const std::size_t> pool,
                                    const DbDensityIndex &density_idx) {
    RangeCoverageModel model;
    const std::size_t pool_size = pool.size();
    model.candidate_covers.resize(pool_size);
    if (pool_size == 0)
        return model;

    constexpr double kHalfBox = 500.0;     // 1km box half-width
    constexpr double kHalfTime = 302400.0; // 7-day window half-width
    const double cell_m = density_idx.cell_m();
    const double inv_cell = 1.0 / cell_m;
    const double time_bin = static_cast<double>(density_idx.time_bin_s());
    const double inv_time = 1.0 / time_bin;

    std::unordered_map<std::uint64_t, std::uint16_t> key_to_unit_id;
    model.unit_weights.reserve(128);
    double weight_sum = 0.0;

    for (std::size_t pool_pos = 0; pool_pos < pool_size; ++pool_pos) {
        const std::size_t pt_idx = pool[pool_pos];
        const double px = traj.x[pt_idx];
        const double py = traj.y[pt_idx];
        const double pt = static_cast<double>(traj.t[pt_idx]);
        // Exact integer cell bounds satisfying |p - (c + 0.5) * step| <= kHalf
        const auto cx_lo = static_cast<std::int32_t>(std::ceil((px - kHalfBox) * inv_cell - 0.5));
        const auto cx_hi = static_cast<std::int32_t>(std::floor((px + kHalfBox) * inv_cell - 0.5));
        const auto cy_lo = static_cast<std::int32_t>(std::ceil((py - kHalfBox) * inv_cell - 0.5));
        const auto cy_hi = static_cast<std::int32_t>(std::floor((py + kHalfBox) * inv_cell - 0.5));
        const auto ct_lo = static_cast<std::int32_t>(std::ceil((pt - kHalfTime) * inv_time - 0.5));
        const auto ct_hi = static_cast<std::int32_t>(std::floor((pt + kHalfTime) * inv_time - 0.5));

        auto &covers = model.candidate_covers[pool_pos];
        for (std::int32_t cell_x = cx_lo; cell_x <= cx_hi; ++cell_x) {
            for (std::int32_t cell_y = cy_lo; cell_y <= cy_hi; ++cell_y) {
                for (std::int32_t cell_t = ct_lo; cell_t <= ct_hi; ++cell_t) {
                    const double center_x = (static_cast<double>(cell_x) + 0.5) * cell_m;
                    const double center_y = (static_cast<double>(cell_y) + 0.5) * cell_m;
                    const double center_t = (static_cast<double>(cell_t) + 0.5) * time_bin;
                    if (std::abs(px - center_x) > kHalfBox ||
                        std::abs(py - center_y) > kHalfBox ||
                        std::abs(pt - center_t) > kHalfTime)
                        continue;

                    const std::uint64_t key = DbDensityIndex::pack_3d(cell_x, cell_y, cell_t);
                    auto it = key_to_unit_id.find(key);
                    std::uint16_t unit_id = 0;
                    if (it == key_to_unit_id.end()) {
                        const auto stats = density_idx.cell_stats(cell_x, cell_y, cell_t);
                        if (stats.point_count == 0 || model.unit_weights.size() >= 2500)
                            continue;
                        double neighbor_trajs = static_cast<double>(stats.traj_count);
                        for (int dx = -1; dx <= 1; ++dx)
                            for (int dy = -1; dy <= 1; ++dy)
                                if (dx != 0 || dy != 0)
                                    neighbor_trajs +=
                                        0.35 *
                                        static_cast<double>(
                                            density_idx.cell_stats(cell_x + dx, cell_y + dy, cell_t)
                                                .traj_count);
                        const double accept_prob =
                            neighbor_trajs < 3.0 ? 0.05 : std::min(1.0, neighbor_trajs / 20.0);
                        const double cell_weight = static_cast<double>(stats.point_count) *
                                                   accept_prob / std::max(3.0, neighbor_trajs);
                        unit_id = static_cast<std::uint16_t>(model.unit_weights.size());
                        key_to_unit_id.emplace(key, unit_id);
                        model.unit_weights.push_back(cell_weight);
                        weight_sum += cell_weight;
                    } else {
                        unit_id = it->second;
                    }
                    covers.push_back(unit_id);
                }
            }
        }
    }

    model.total_weight = std::max(1e-6, weight_sum);
    return model;
}

RangeCoverageModel build_blended_range_model(const core::TrajectoryView &traj,
                                             std::span<const std::size_t> pool,
                                             const DbDensityIndex &density_idx,
                                             std::span<const SampledRangeBox> boxes,
                                             double uncovered_frac) {
    RangeCoverageModel model;
    const std::size_t pool_size = pool.size();
    model.candidate_covers.resize(pool_size);
    if (pool_size == 0 || boxes.empty())
        return model; // Zero uncovered valid [5, 20] Range boxes -> spend budget on Geo & Where-At!

    // 1. 20% continuous 333m x 3.5d cell grid floor (never runs dry at 2% budget)
    model = build_cell_model(traj, pool, density_idx);
    const double cell_scale = 0.20 / model.total_weight;
    for (double &weight : model.unit_weights)
        weight *= cell_scale;

    // 2. 80% exact sampled 1km x 7d uncovered Range boxes scaled by true uncovered fraction
    //    (so a trip already 90% covered by endpoints doesn't steal points from Where-At / Join!)
    const std::size_t first_box_unit_id = model.unit_weights.size();
    const std::size_t num_boxes = std::min<std::size_t>(boxes.size(), 65000 - first_box_unit_id);
    double box_weight_sum = 0.0;
    for (std::size_t box_idx = 0; box_idx < num_boxes; ++box_idx) {
        const auto &sampled_box = boxes[box_idx];
        const auto &box = sampled_box.box;
        bool is_covered = false;
        const auto unit_id = static_cast<std::uint16_t>(model.unit_weights.size());
        for (std::size_t pool_pos = 0; pool_pos < pool_size; ++pool_pos) {
            const std::size_t pt_idx = pool[pool_pos];
            const std::int64_t pt_t = traj.t[pt_idx];
            if (pt_t < box.t_min)
                continue;
            if (pt_t > box.t_max)
                break;
            const double pt_x = traj.x[pt_idx], pt_y = traj.y[pt_idx];
            if (pt_x >= box.x_min && pt_x <= box.x_max && pt_y >= box.y_min && pt_y <= box.y_max) {
                model.candidate_covers[pool_pos].push_back(unit_id);
                is_covered = true;
            }
        }
        if (is_covered) {
            model.unit_weights.push_back(sampled_box.weight);
            box_weight_sum += sampled_box.weight;
        }
    }
    if (box_weight_sum > 1e-9) {
        const double box_scale = (0.80 * std::clamp(uncovered_frac, 0.0, 1.0)) / box_weight_sum;
        for (std::size_t idx = first_box_unit_id; idx < model.unit_weights.size(); ++idx)
            model.unit_weights[idx] *= box_scale;
    }
    model.total_weight = 1.0;
    return model;
}

struct SegmentScores {
    double shape_arc_sum = 0.0; // exact trapezoidal arc-weighted shape in [0, seg_arc]
    double ped_l1_sum = 0.0;    // exact trapezoidal arc-weighted L1 perpendicular error (m * m)
    double max_dev = 0.0;       // exact max perpendicular segment distance (m)
    double wa_sum = 0.0;        // sum of ramp_unit over probes in [t_a, t_b)
};

SegmentScores eval_segment_scores(const core::TrajectoryView &traj,
                                  std::span<const std::size_t> pool,
                                  std::span<const double> arc_pool, std::span<const Probe> probes,
                                  std::span<const std::size_t> probe_start_of_cand,
                                  std::size_t start_pos, std::size_t end_pos) {
    SegmentScores scores;
    const std::size_t start_idx = pool[start_pos], end_idx = pool[end_pos];
    const double ax = traj.x[start_idx], ay = traj.y[start_idx];
    const double bx = traj.x[end_idx], by = traj.y[end_idx];
    const double seg_dx = bx - ax, seg_dy = by - ay;

    // 1. Exact perpendicular segment distance + trapezoidal arc-weighted shape & L1 error
    double max_dev = 0.0;
    double arc_sum = 0.0;
    double l1_sum = 0.0;
    double prev_ramp = 1.0;
    double prev_dist = 0.0;
    for (std::size_t pool_pos = start_pos + 1; pool_pos <= end_pos; ++pool_pos) {
        const double arc_step = arc_pool[pool_pos] - arc_pool[pool_pos - 1];
        double ramp = 1.0;
        double dist = 0.0;
        if (pool_pos < end_pos) {
            const std::size_t pt_idx = pool[pool_pos];
            dist = std::sqrt(selection::distance::squared_segment_distance(
                traj.x[pt_idx], traj.y[pt_idx], ax, ay, bx, by));
            if (dist > max_dev)
                max_dev = dist;
            ramp = ramp_unit(dist);
        }
        if (arc_step > 0.0) {
            arc_sum += arc_step * 0.5 * (prev_ramp + ramp);
            l1_sum += arc_step * 0.5 * (prev_dist + dist);
        }
        prev_ramp = ramp;
        prev_dist = dist;
    }
    scores.shape_arc_sum = arc_sum;
    scores.ped_l1_sum = l1_sum;
    scores.max_dev = max_dev;

    // 2. Where-At over probes in [probe_start_of_cand[start_pos], probe_start_of_cand[end_pos])
    const std::size_t probe_begin = probe_start_of_cand[start_pos];
    const std::size_t probe_end = probe_start_of_cand[end_pos];
    const double start_t = static_cast<double>(traj.t[start_idx]);
    const double dt = static_cast<double>(traj.t[end_idx] - traj.t[start_idx]);
    const double inv_dt = dt > 0.0 ? 1.0 / dt : 0.0;
    double where_at_sum = 0.0;
    for (std::size_t probe_idx = probe_begin; probe_idx < probe_end; ++probe_idx) {
        const double frac = std::clamp((probes[probe_idx].t - start_t) * inv_dt, 0.0, 1.0);
        const double interp_x = ax + frac * seg_dx;
        const double interp_y = ay + frac * seg_dy;
        const double dist =
            std::hypot(interp_x - probes[probe_idx].x, interp_y - probes[probe_idx].y);
        where_at_sum += ramp_unit(dist);
    }
    scores.wa_sum = where_at_sum;
    return scores;
}

} // namespace

std::vector<std::size_t>
submodular_simplify_on_pool(const core::TrajectoryView &traj, std::span<const std::size_t> pool,
                            std::size_t budget, const SubmodularWeights &weights,
                            const DbDensityIndex &density_idx,
                            const std::vector<SampledRangeBox> *uncov_boxes, double uncov_frac) {
    const std::size_t traj_len = traj.size();
    const std::size_t pool_size = pool.size();
    if (budget >= traj_len) {
        std::vector<std::size_t> all_indices(traj_len);
        for (std::size_t idx = 0; idx < traj_len; ++idx)
            all_indices[idx] = idx;
        return all_indices;
    }
    budget = std::clamp(budget, std::size_t{2}, traj_len);
    if (budget == 2 || pool_size <= 2)
        return {0, traj_len - 1};
    if (budget >= pool_size) {
        std::vector<std::size_t> kept_indices(pool.begin(), pool.end());
        if (kept_indices.size() < budget) {
            std::vector<std::uint8_t> is_used(traj_len, 0);
            for (std::size_t idx : kept_indices)
                is_used[idx] = 1;
            for (std::size_t idx = 1; idx + 1 < traj_len && kept_indices.size() < budget; ++idx) {
                if (!is_used[idx]) {
                    is_used[idx] = 1;
                    kept_indices.push_back(idx);
                }
            }
            std::ranges::sort(kept_indices);
        }
        return kept_indices;
    }

    const auto full_arc = compute_arc_length(traj);
    std::vector<double> arc_pool(pool_size);
    for (std::size_t pool_pos = 0; pool_pos < pool_size; ++pool_pos)
        arc_pool[pool_pos] = full_arc[pool[pool_pos]];
    const double inv_total_arc = 1.0 / std::max(1.0, arc_pool.back());

    const auto probes = build_probes(traj, 64);
    const double inv_probes = probes.empty() ? 0.0 : 1.0 / static_cast<double>(probes.size());
    std::vector<std::size_t> probe_start(pool_size, probes.size());
    {
        std::size_t probe_idx = 0;
        for (std::size_t pool_pos = 0; pool_pos < pool_size; ++pool_pos) {
            const double cand_t = static_cast<double>(traj.t[pool[pool_pos]]);
            while (probe_idx < probes.size() && probes[probe_idx].t < cand_t)
                ++probe_idx;
            probe_start[pool_pos] = probe_idx;
        }
        probe_start[pool_size - 1] = probes.size();
    }

    const auto range_model =
        (uncov_boxes != nullptr)
            ? build_blended_range_model(traj, pool, density_idx, *uncov_boxes, uncov_frac)
            : build_cell_model(traj, pool, density_idx);
    const std::size_t num_units = range_model.unit_weights.size();
    std::vector<std::uint16_t> cover_counts(num_units, 0);

    auto add_cover = [&](std::size_t pool_pos) {
        for (std::uint16_t unit_id : range_model.candidate_covers[pool_pos])
            cover_counts[unit_id] += 1;
    };
    auto remove_cover = [&](std::size_t pool_pos) {
        for (std::uint16_t unit_id : range_model.candidate_covers[pool_pos])
            cover_counts[unit_id] -= 1;
    };
    auto range_gain_if_added = [&](std::size_t pool_pos) -> double {
        double gain = 0.0;
        for (std::uint16_t unit_id : range_model.candidate_covers[pool_pos])
            if (cover_counts[unit_id] == 0)
                gain += range_model.unit_weights[unit_id];
        return gain / range_model.total_weight;
    };

    // Active kept pool positions (kept sorted)
    std::vector<std::size_t> kept_positions;
    kept_positions.reserve(budget);
    kept_positions.push_back(0);
    kept_positions.push_back(pool_size - 1);
    add_cover(0);
    add_cover(pool_size - 1);

    // Track current segment scores for each interval (kept_positions[seg], kept_positions[seg + 1])
    std::vector<SegmentScores> segment_scores;
    segment_scores.reserve(budget);
    segment_scores.push_back(
        eval_segment_scores(traj, pool, arc_pool, probes, probe_start, 0, pool_size - 1));

    // Greedy submodular insertion: at each step pick the candidate maximizing
    //   w_geo * geo_gain + w_wa * where_at_gain + w_rng * range_gain
    // Scan up to 8 candidates inside each active segment: max-PED, max-SED, highest uncovered-Range,
    // plus 5 uniformly spaced interior positions.
    while (kept_positions.size() < budget) {
        double best_gain = -1e18;
        std::size_t best_seg = 0;
        std::size_t best_pos = 0;

        // Current whole-trajectory max perpendicular deviation (for L_inf Shape term)
        double global_max_dev = 0.0;
        std::size_t max_dev_seg = 0;
        for (std::size_t seg = 0; seg < segment_scores.size(); ++seg) {
            if (segment_scores[seg].max_dev > global_max_dev) {
                global_max_dev = segment_scores[seg].max_dev;
                max_dev_seg = seg;
            }
        }
        double second_max_dev = 0.0;
        for (std::size_t seg = 0; seg < segment_scores.size(); ++seg)
            if (seg != max_dev_seg && segment_scores[seg].max_dev > second_max_dev)
                second_max_dev = segment_scores[seg].max_dev;

        for (std::size_t seg = 0; seg < segment_scores.size(); ++seg) {
            const std::size_t start_pos = kept_positions[seg];
            const std::size_t end_pos = kept_positions[seg + 1];
            if (end_pos <= start_pos + 1)
                continue;
            const auto &base_scores = segment_scores[seg];
            const bool is_worst_seg = (seg == max_dev_seg);

            // Pick a compact candidate set inside (start_pos, end_pos):
            // (a) max-PED point, (b) max-SED point, (c) highest uncovered-Range point, (d) 5 uniform points
            const std::size_t start_idx = pool[start_pos], end_idx = pool[end_pos];
            const double ax = traj.x[start_idx], ay = traj.y[start_idx];
            const double bx = traj.x[end_idx], by = traj.y[end_idx];
            const double start_t = static_cast<double>(traj.t[start_idx]);
            const double dt = static_cast<double>(traj.t[end_idx] - traj.t[start_idx]);
            const double inv_dt = dt > 0.0 ? 1.0 / dt : 0.0;
            std::size_t arg_dev = start_pos + 1;
            std::size_t arg_sed = start_pos + 1;
            std::size_t arg_rng = start_pos + 1;
            double max_dev_sq = -1.0, max_sed_sq = -1.0, max_rng = -1.0;

            const std::size_t span = end_pos - start_pos - 1;
            for (std::size_t cand_pos = start_pos + 1; cand_pos < end_pos; ++cand_pos) {
                const std::size_t pt_idx = pool[cand_pos];
                const double perp_dist_sq = selection::distance::squared_segment_distance(
                    traj.x[pt_idx], traj.y[pt_idx], ax, ay, bx, by);
                if (perp_dist_sq > max_dev_sq) {
                    max_dev_sq = perp_dist_sq;
                    arg_dev = cand_pos;
                }
                const double frac = (static_cast<double>(traj.t[pt_idx]) - start_t) * inv_dt;
                const double sed_dx = ax + frac * (bx - ax) - traj.x[pt_idx];
                const double sed_dy = ay + frac * (by - ay) - traj.y[pt_idx];
                const double sed_dist_sq = sed_dx * sed_dx + sed_dy * sed_dy;
                if (sed_dist_sq > max_sed_sq) {
                    max_sed_sq = sed_dist_sq;
                    arg_sed = cand_pos;
                }
                const double range_gain = range_gain_if_added(cand_pos);
                if (range_gain > max_rng) {
                    max_rng = range_gain;
                    arg_rng = cand_pos;
                }
            }

            std::array<std::size_t, 8> candidates{};
            std::size_t num_candidates = 0;
            candidates[num_candidates++] = arg_dev;
            candidates[num_candidates++] = arg_sed;
            candidates[num_candidates++] = arg_rng;
            const std::size_t num_uniform = std::min<std::size_t>(5, span);
            for (std::size_t step_idx = 1; step_idx <= num_uniform; ++step_idx)
                candidates[num_candidates++] =
                    start_pos + (step_idx * (end_pos - start_pos)) / (num_uniform + 1);
            std::sort(candidates.begin(),
                      candidates.begin() + static_cast<std::ptrdiff_t>(num_candidates));
            num_candidates = static_cast<std::size_t>(
                std::unique(candidates.begin(),
                            candidates.begin() + static_cast<std::ptrdiff_t>(num_candidates)) -
                candidates.begin());

            for (std::size_t cand_idx = 0; cand_idx < num_candidates; ++cand_idx) {
                const std::size_t cand_pos = candidates[cand_idx];
                if (cand_pos <= start_pos || cand_pos >= end_pos)
                    continue;
                const auto left_scores =
                    eval_segment_scores(traj, pool, arc_pool, probes, probe_start, start_pos, cand_pos);
                const auto right_scores =
                    eval_segment_scores(traj, pool, arc_pool, probes, probe_start, cand_pos, end_pos);

                // Exact RDP point deviation (never vanishes on S-curves!) + L1 area drop + arc shape
                const std::size_t pt_idx = pool[cand_pos];
                const double perp_dist = std::sqrt(selection::distance::squared_segment_distance(
                    traj.x[pt_idx], traj.y[pt_idx], ax, ay, bx, by));
                const double l1_drop =
                    std::max(0.0,
                             base_scores.ped_l1_sum -
                                 (left_scores.ped_l1_sum + right_scores.ped_l1_sum)) *
                    inv_total_arc;
                const double arc_gain =
                    (left_scores.shape_arc_sum + right_scores.shape_arc_sum -
                     base_scores.shape_arc_sum) *
                    inv_total_arc;
                const double new_seg_max = std::max(left_scores.max_dev, right_scores.max_dev);
                double global_max_drop = 0.0;
                if (is_worst_seg) {
                    const double new_global_max = std::max(second_max_dev, new_seg_max);
                    global_max_drop = std::max(0.0, global_max_dev - new_global_max) / 150.0;
                }
                const double geo_gain = 0.50 * (perp_dist / 150.0) + 0.30 * (l1_drop / 35.0) +
                                        0.20 * global_max_drop + 0.40 * arc_gain;

                const double where_at_gain =
                    (left_scores.wa_sum + right_scores.wa_sum - base_scores.wa_sum) * inv_probes;
                const double range_gain = range_gain_if_added(cand_pos);

                const double gain = weights.w_geo * geo_gain + weights.w_wa * where_at_gain +
                                    weights.w_rng * range_gain;
                if (gain > best_gain) {
                    best_gain = gain;
                    best_seg = seg;
                    best_pos = cand_pos;
                }
            }
        }

        if (best_pos == 0)
            break;

        const std::size_t start_pos = kept_positions[best_seg];
        const std::size_t end_pos = kept_positions[best_seg + 1];
        kept_positions.insert(kept_positions.begin() + static_cast<std::ptrdiff_t>(best_seg + 1),
                              best_pos);
        add_cover(best_pos);
        segment_scores[best_seg] =
            eval_segment_scores(traj, pool, arc_pool, probes, probe_start, start_pos, best_pos);
        segment_scores.insert(
            segment_scores.begin() + static_cast<std::ptrdiff_t>(best_seg + 1),
            eval_segment_scores(traj, pool, arc_pool, probes, probe_start, best_pos, end_pos));
    }

    // Fast Red-Black Free-Knot Polish optimizing the EXACT same 3-term objective
    for (int pass = 0; pass < weights.rb_passes; ++pass) {
        bool changed = false;
        for (int parity = 1; parity <= 2; ++parity) {
            for (std::size_t knot_idx = static_cast<std::size_t>(parity);
                 knot_idx + 1 < kept_positions.size(); knot_idx += 2) {
                const std::size_t left_pos = kept_positions[knot_idx - 1];
                const std::size_t right_pos = kept_positions[knot_idx + 1];
                if (right_pos <= left_pos + 2)
                    continue;
                const std::size_t original_pos = kept_positions[knot_idx];

                // Compute max_dev across all OTHER segments for exact L_inf
                double other_max_dev = 0.0;
                for (std::size_t seg = 0; seg < segment_scores.size(); ++seg)
                    if (seg != knot_idx - 1 && seg != knot_idx &&
                        segment_scores[seg].max_dev > other_max_dev)
                        other_max_dev = segment_scores[seg].max_dev;

                remove_cover(original_pos);

                auto eval_knot_position = [&](std::size_t cand_pos) -> double {
                    const auto left_scores = eval_segment_scores(
                        traj, pool, arc_pool, probes, probe_start, left_pos, cand_pos);
                    const auto right_scores = eval_segment_scores(
                        traj, pool, arc_pool, probes, probe_start, cand_pos, right_pos);
                    const double geo_arc =
                        (left_scores.shape_arc_sum + right_scores.shape_arc_sum) * inv_total_arc;
                    const double geo_l1 =
                        (left_scores.ped_l1_sum + right_scores.ped_l1_sum) * inv_total_arc;
                    const double new_seg_max = std::max(left_scores.max_dev, right_scores.max_dev);
                    const double new_global_max = std::max(other_max_dev, new_seg_max);
                    const double geo_score = 0.40 * geo_arc - 0.35 * (geo_l1 / 35.0) -
                                             0.35 * (new_seg_max / 150.0) -
                                             0.25 * (new_global_max / 150.0);
                    const double where_at_score =
                        (left_scores.wa_sum + right_scores.wa_sum) * inv_probes;
                    const double range_score = range_gain_if_added(cand_pos);
                    return weights.w_geo * geo_score + weights.w_wa * where_at_score +
                           weights.w_rng * range_score;
                };

                std::size_t best_pos = original_pos;
                double best_score = eval_knot_position(original_pos);

                // Coarse scan across (left_pos, right_pos) + local halving
                const std::size_t span = right_pos - left_pos - 1;
                const std::size_t num_coarse = std::min<std::size_t>(10, span);
                for (std::size_t step_idx = 1; step_idx <= num_coarse; ++step_idx) {
                    const std::size_t cand_pos =
                        left_pos + (step_idx * (right_pos - left_pos)) / (num_coarse + 1);
                    if (cand_pos <= left_pos || cand_pos >= right_pos || cand_pos == best_pos)
                        continue;
                    const double score = eval_knot_position(cand_pos);
                    if (score > best_score + 1e-9) {
                        best_score = score;
                        best_pos = cand_pos;
                    }
                }
                std::size_t step = std::max<std::size_t>(1, span / (num_coarse + 1));
                while (step >= 1) {
                    bool improved = false;
                    if (best_pos > left_pos + step) {
                        const double score = eval_knot_position(best_pos - step);
                        if (score > best_score + 1e-9) {
                            best_score = score;
                            best_pos -= step;
                            improved = true;
                        }
                    }
                    if (best_pos + step < right_pos) {
                        const double score = eval_knot_position(best_pos + step);
                        if (score > best_score + 1e-9) {
                            best_score = score;
                            best_pos += step;
                            improved = true;
                        }
                    }
                    if (!improved)
                        step /= 2;
                }

                kept_positions[knot_idx] = best_pos;
                add_cover(best_pos);
                if (best_pos != original_pos) {
                    segment_scores[knot_idx - 1] = eval_segment_scores(
                        traj, pool, arc_pool, probes, probe_start, left_pos, best_pos);
                    segment_scores[knot_idx] = eval_segment_scores(
                        traj, pool, arc_pool, probes, probe_start, best_pos, right_pos);
                    changed = true;
                }
            }
        }
        if (!changed)
            break;
    }

    std::vector<std::size_t> kept_indices(kept_positions.size());
    for (std::size_t idx = 0; idx < kept_positions.size(); ++idx)
        kept_indices[idx] = pool[kept_positions[idx]];
    return kept_indices;
}

} // namespace student
