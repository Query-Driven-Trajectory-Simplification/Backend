#include "backend/engine/teacher/greedy.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>

namespace teacher {

namespace {

// Smallest gain worth keeping a point for; below it greedy stops and fill takes over
constexpr double kMinGain = 1e-15;

} // namespace

Greedy::Greedy(const Objective &objective, std::size_t i)
    : m_trajectory(objective.db().trajectory(i)),
      m_min_tolerance(objective.config().min_tolerance_m),
      m_max_tolerance(objective.config().max_tolerance_m) {
    const auto &trajectory = m_trajectory;
    const std::size_t n = trajectory.size();
    const auto &workloads = objective.workloads();

    const double range_scale = objective.scale(Objective::Range);
    std::vector<std::uint32_t> count(n, 0);
    for (const auto &[b, weight] : objective.boxes(i)) {
        const auto &box = workloads.range.queries[b].box;
        std::vector<std::uint32_t> inside;
        // t is sorted, so only the box's time window is scanned
        for (auto k = static_cast<std::size_t>(std::ranges::lower_bound(trajectory.t, box.t_min) -
                                               trajectory.t.begin());
             k < n && trajectory.t[k] <= box.t_max; ++k)
            if (trajectory.x[k] >= box.x_min && trajectory.x[k] <= box.x_max &&
                trajectory.y[k] >= box.y_min && trajectory.y[k] <= box.y_max) {
                inside.push_back(static_cast<std::uint32_t>(k));
                ++count[k];
            }
        m_box_points.push_back(std::move(inside));
        m_box_weight.push_back(range_scale * weight);
    }
    m_offsets.assign(n + 1, 0);
    for (std::size_t k = 0; k < n; ++k)
        m_offsets[k + 1] = m_offsets[k] + count[k];
    m_point_boxes.resize(m_offsets[n]);
    std::vector<std::uint64_t> at(m_offsets.begin(), m_offsets.end() - 1);
    for (std::size_t b = 0; b < m_box_points.size(); ++b)
        for (const auto k : m_box_points[b])
            m_point_boxes[at[k]++] = static_cast<std::uint32_t>(b);

    const auto probes = objective.probes(i);
    std::vector<std::uint32_t> order(probes.begin(), probes.end());
    std::ranges::sort(order, [&](auto a, auto b) {
        return workloads.where_at.queries[a].t < workloads.where_at.queries[b].t;
    });
    for (const auto k : order) {
        const auto &probe = workloads.where_at.queries[k];
        m_probe_t.push_back(probe.t);
        m_probe_x.push_back(probe.reference.x);
        m_probe_y.push_back(probe.reference.y);
    }
    if (!order.empty())
        m_probe_weight = objective.scale(Objective::WhereAt) * objective.where_at_weight() /
                         static_cast<double>(order.size());
}

std::vector<std::size_t> Greedy::run(std::size_t budget, std::span<const std::size_t> seed,
                                     std::span<const std::size_t> fill) const {
    auto kept = sequence(budget, seed, fill);
    std::ranges::sort(kept);
    return kept;
}

std::vector<std::size_t> Greedy::sequence(std::size_t limit, std::span<const std::size_t> seed,
                                          std::span<const std::size_t> fill) const {
    const auto &trajectory = m_trajectory;
    const std::size_t n = trajectory.size();
    const std::size_t budget = std::min(limit, n);

    // Squared errors that put a probe clearly past either end of the ramp, by a margin far above
    // the rounding of dx * dx + dy * dy
    const double far = m_max_tolerance * m_max_tolerance * (1.0 + 1e-12);
    const double near = m_min_tolerance * m_min_tolerance * (1.0 - 1e-12);
    // Where-at score of probe p on the segment from kept point a to kept point c
    auto score_on = [&](std::size_t p, std::size_t from, std::size_t to) {
        const double t_from = static_cast<double>(trajectory.t[from]),
                     t_to = static_cast<double>(trajectory.t[to]);
        const double fraction =
            t_to > t_from ? std::clamp((m_probe_t[p] - t_from) / (t_to - t_from), 0.0, 1.0) : 0.0;
        const double x = trajectory.x[from] + fraction * (trajectory.x[to] - trajectory.x[from]);
        const double y = trajectory.y[from] + fraction * (trajectory.y[to] - trajectory.y[from]);
        const double dx = x - m_probe_x[p], dy = y - m_probe_y[p];
        // Clearly outside the ramp it is 0 or 1 exactly, without the square root
        const double squared = dx * dx + dy * dy;
        if (squared >= far)
            return 0.0;
        if (squared <= near)
            return 1.0;
        return ramp(std::hypot(dx, dy), m_min_tolerance, m_max_tolerance);
    };
    // Probes from t_a up to t_c, the last segment including its end
    auto probes_between = [&](std::size_t from, std::size_t to) {
        const auto lo = static_cast<std::size_t>(
            std::ranges::lower_bound(m_probe_t, static_cast<double>(trajectory.t[from])) -
            m_probe_t.begin());
        const auto hi =
            to == n - 1
                ? m_probe_t.size()
                : static_cast<std::size_t>(
                      std::ranges::lower_bound(m_probe_t, static_cast<double>(trajectory.t[to])) -
                      m_probe_t.begin());
        return std::pair{lo, hi};
    };

    std::vector<char> kept(n, 0), covered(m_box_points.size(), 0);
    std::vector<double> range_gain(n, 0.0), where_at_gain(n, 0.0);
    for (std::size_t box = 0; box < m_box_points.size(); ++box)
        for (const auto k : m_box_points[box])
            range_gain[k] += m_box_weight[box];
    std::set<std::size_t> order;
    std::vector<std::size_t> sequence;
    std::size_t count = 0;
    auto keep = [&](std::size_t j) {
        if (kept[j])
            return;
        kept[j] = 1;
        order.insert(j);
        sequence.push_back(j);
        ++count;
        for (auto e = m_offsets[j]; e < m_offsets[j + 1]; ++e) {
            const auto box = m_point_boxes[e];
            if (covered[box])
                continue;
            covered[box] = 1;
            for (const auto k : m_box_points[box])
                range_gain[k] -= m_box_weight[box];
        }
    };
    // Where-at gain of every point strictly between kept neighbours a and c
    std::vector<double> current_scores(m_probe_t.size());
    auto refresh = [&](std::size_t from, std::size_t to) {
        const auto [lo, hi] = probes_between(from, to);
        for (std::size_t p = lo; p < hi; ++p)
            current_scores[p] = score_on(p, from, to);
        for (std::size_t j = from + 1; j < to; ++j) {
            double gain = 0;
            for (std::size_t p = lo; p < hi; ++p) {
                const double now = current_scores[p];
                const double then = m_probe_t[p] < static_cast<double>(trajectory.t[j])
                                        ? score_on(p, from, j)
                                        : score_on(p, j, to);
                gain += then - now;
            }
            where_at_gain[j] = gain * m_probe_weight;
        }
    };

    keep(0);
    keep(n - 1);
    for (const auto j : seed)
        if (count < budget)
            keep(j);
    if (!m_probe_t.empty() && count < budget)
        for (auto it = order.begin(); std::next(it) != order.end(); ++it)
            refresh(*it, *std::next(it));

    while (count < budget) {
        double best = kMinGain;
        std::size_t pick = n;
        for (std::size_t j = 1; j + 1 < n; ++j)
            if (!kept[j] && range_gain[j] + where_at_gain[j] > best) {
                best = range_gain[j] + where_at_gain[j];
                pick = j;
            }
        if (pick == n)
            break;
        keep(pick);
        if (!m_probe_t.empty() && count < budget) {
            const auto it = order.find(pick);
            refresh(*std::prev(it), pick);
            refresh(pick, *std::next(it));
        }
    }
    for (std::size_t z = 0; z < fill.size() && count < budget; ++z)
        keep(fill[z]);
    for (std::size_t j = 1; j + 1 < n && count < budget; ++j) // only if fill ran short
        keep(j);

    return sequence;
}

// Shared per-kept-set segment, Where-At, and Range-box state for 1-opt refinement and point trades
struct Greedy::LocalContext {
    const Greedy &g;
    std::span<const std::size_t> kept;
    std::size_t n = 0;
    std::size_t b = 0;
    std::size_t segs = 0;
    double far = 0.0;
    double near = 0.0;
    std::vector<char> is_kept;
    std::vector<double> arc;
    std::vector<std::uint32_t> box_hits;
    std::vector<double> live_box_weight;
    std::vector<double> seg_err;
    std::vector<double> seg_wa_score;
    std::vector<std::size_t> worst_pt;
    std::vector<std::size_t> seg_order;

    LocalContext(const Greedy &greedy, std::span<const std::size_t> kept_set,
                 std::span<const double> range_weights, double range_scale,
                 std::span<const double> segments)
        : g(greedy), kept(kept_set), n(g.m_trajectory.size()), b(kept_set.size()), segs(b - 1),
          far(g.m_max_tolerance * g.m_max_tolerance * (1.0 + 1e-12)),
          near(g.m_min_tolerance * g.m_min_tolerance * (1.0 - 1e-12)), is_kept(n, 0), arc(n, 0.0),
          box_hits(g.m_box_points.size(), 0), live_box_weight(g.m_box_points.size(), 0.0),
          seg_err(segs, 0.0), seg_wa_score(segs, 0.0), worst_pt(segs, n), seg_order(segs) {
        const auto &t = g.m_trajectory;
        for (const auto idx : kept)
            is_kept[idx] = 1;
        for (std::size_t j = 0; j + 1 < n; ++j)
            arc[j + 1] =
                arc[j] + (segments.empty() ? std::hypot(t.x[j + 1] - t.x[j], t.y[j + 1] - t.y[j])
                                           : segments[j]);
        for (const auto idx : kept)
            for (auto e = g.m_offsets[idx]; e < g.m_offsets[idx + 1]; ++e)
                ++box_hits[g.m_point_boxes[e]];
        for (std::size_t box = 0; box < g.m_box_points.size(); ++box)
            live_box_weight[box] =
                range_weights.empty() ? g.m_box_weight[box] : range_scale * range_weights[box];
        for (std::size_t r = 0; r < segs; ++r) {
            const auto [sq, w] = seg_deviation(kept[r], kept[r + 1]);
            seg_err[r] = sq;
            worst_pt[r] = w;
            if (g.m_probe_weight > 0.0)
                seg_wa_score[r] = seg_wa(kept[r], kept[r + 1]);
        }
        std::iota(seg_order.begin(), seg_order.end(), 0);
        std::ranges::sort(seg_order, [&](std::size_t a, std::size_t c) {
            return seg_err[a] > seg_err[c] || (seg_err[a] == seg_err[c] && a < c);
        });
    }

    double score_on(std::size_t p, std::size_t from, std::size_t to) const {
        const auto &t = g.m_trajectory;
        const double t_from = static_cast<double>(t.t[from]), t_to = static_cast<double>(t.t[to]);
        const double fraction =
            t_to > t_from ? std::clamp((g.m_probe_t[p] - t_from) / (t_to - t_from), 0.0, 1.0) : 0.0;
        const double x = t.x[from] + fraction * (t.x[to] - t.x[from]);
        const double y = t.y[from] + fraction * (t.y[to] - t.y[from]);
        const double dx = x - g.m_probe_x[p], dy = y - g.m_probe_y[p];
        const double squared = dx * dx + dy * dy;
        if (squared >= far)
            return 0.0;
        if (squared <= near)
            return 1.0;
        return ramp(std::hypot(dx, dy), g.m_min_tolerance, g.m_max_tolerance);
    }

    std::pair<std::size_t, std::size_t> probes_between(std::size_t from, std::size_t to) const {
        if (g.m_probe_t.empty())
            return {0, 0};
        const auto &t = g.m_trajectory;
        const auto lo = static_cast<std::size_t>(
            std::ranges::lower_bound(g.m_probe_t, static_cast<double>(t.t[from])) -
            g.m_probe_t.begin());
        const auto hi =
            to == n - 1 ? g.m_probe_t.size()
                        : static_cast<std::size_t>(
                              std::ranges::lower_bound(g.m_probe_t, static_cast<double>(t.t[to])) -
                              g.m_probe_t.begin());
        return {lo, hi};
    }

    double seg_wa(std::size_t from, std::size_t to) const {
        const auto [lo, hi] = probes_between(from, to);
        double sum = 0.0;
        for (std::size_t p = lo; p < hi; ++p)
            sum += score_on(p, from, to);
        return sum;
    }

    std::pair<double, std::size_t> seg_deviation(std::size_t from, std::size_t to) const {
        const auto &t = g.m_trajectory;
        const double length = arc[to] - arc[from];
        const double ax = t.x[from], ay = t.y[from];
        const double dx = t.x[to] - ax, dy = t.y[to] - ay;
        double max_sq = 0.0;
        std::size_t worst = from;
        for (std::size_t j = from + 1; j < to; ++j) {
            const double u = length > 0.0 ? (arc[j] - arc[from]) / length : 0.0;
            const double ex = t.x[j] - (ax + u * dx);
            const double ey = t.y[j] - (ay + u * dy);
            const double sq = ex * ex + ey * ey;
            if (sq > max_sq) {
                max_sq = sq;
                worst = j;
            }
        }
        return {max_sq, worst};
    }

    std::size_t rdp_peak(std::size_t from, std::size_t to) const {
        const auto &t = g.m_trajectory;
        const double ax = t.x[from], ay = t.y[from];
        const double dx = t.x[to] - ax, dy = t.y[to] - ay;
        const double len_sq = dx * dx + dy * dy;
        double best_sq = -1.0;
        std::size_t best_j = from + 1;
        for (std::size_t j = from + 1; j < to; ++j) {
            const double px = t.x[j] - ax, py = t.y[j] - ay;
            double sq;
            if (len_sq == 0.0) {
                sq = px * px + py * py;
            } else {
                const double u = std::clamp((px * dx + py * dy) / len_sq, 0.0, 1.0);
                const double ex = px - u * dx, ey = py - u * dy;
                sq = ex * ex + ey * ey;
            }
            if (sq > best_sq) {
                best_sq = sq;
                best_j = j;
            }
        }
        return best_j;
    }

    std::size_t arc_mid(std::size_t from, std::size_t to) const {
        const double target = 0.5 * (arc[from] + arc[to]);
        const auto it = std::lower_bound(arc.begin() + static_cast<std::ptrdiff_t>(from + 1),
                                         arc.begin() + static_cast<std::ptrdiff_t>(to), target);
        auto j = static_cast<std::size_t>(it - arc.begin());
        if (j >= to)
            j = to - 1;
        return j;
    }

    double eval_rng(std::size_t j) const {
        double gain = 0.0;
        for (auto e = g.m_offsets[j]; e < g.m_offsets[j + 1]; ++e)
            if (box_hits[g.m_point_boxes[e]] == 0)
                gain += live_box_weight[g.m_point_boxes[e]];
        return gain;
    }

    // Up to 4 cheapest droppable interior knot positions m in [1, b - 2]
    std::vector<std::size_t> drop_candidates() const {
        if (b < 3)
            return {};
        struct DropInfo {
            std::size_t m;
            double local_loss;
            double merged_err_sq;
        };
        std::vector<DropInfo> drops;
        drops.reserve(b - 2);
        for (std::size_t m = 1; m + 1 < b; ++m) {
            double rng_loss = 0.0;
            for (auto e = g.m_offsets[kept[m]]; e < g.m_offsets[kept[m] + 1]; ++e)
                if (box_hits[g.m_point_boxes[e]] == 1)
                    rng_loss += live_box_weight[g.m_point_boxes[e]];
            double wa_loss = 0.0;
            if (g.m_probe_weight > 0.0)
                wa_loss =
                    (seg_wa_score[m - 1] + seg_wa_score[m] - seg_wa(kept[m - 1], kept[m + 1])) *
                    g.m_probe_weight;
            drops.push_back({m, rng_loss + wa_loss, seg_deviation(kept[m - 1], kept[m + 1]).first});
        }
        std::vector<std::size_t> drop_set;
        auto add_drop = [&](std::size_t m) {
            if (std::ranges::find(drop_set, m) == drop_set.end())
                drop_set.push_back(m);
        };
        auto by_local = drops;
        std::ranges::sort(by_local, [](const DropInfo &a, const DropInfo &c) {
            return a.local_loss < c.local_loss || (a.local_loss == c.local_loss && a.m < c.m);
        });
        for (std::size_t k = 0; k < std::min<std::size_t>(2, by_local.size()); ++k)
            add_drop(by_local[k].m);
        auto by_err = drops;
        std::ranges::sort(by_err, [](const DropInfo &a, const DropInfo &c) {
            return a.merged_err_sq < c.merged_err_sq ||
                   (a.merged_err_sq == c.merged_err_sq && a.m < c.m);
        });
        for (std::size_t k = 0; k < std::min<std::size_t>(2, by_err.size()); ++k)
            add_drop(by_err[k].m);
        return drop_set;
    }

    // Up to 6 top global unkept insertion points (Fréchet + Range + Where-At)
    std::vector<std::size_t> add_candidates() const {
        if (b >= n)
            return {};
        const auto &t = g.m_trajectory;
        std::vector<std::size_t> add_set;
        auto add_insert = [&](std::size_t q) {
            if (q > 0 && q + 1 < n && !is_kept[q] && std::ranges::find(add_set, q) == add_set.end())
                add_set.push_back(q);
        };
        for (std::size_t k = 0; k < std::min<std::size_t>(2, seg_order.size()); ++k) {
            const std::size_t r = seg_order[k];
            if (seg_err[r] > 0.0 && worst_pt[r] > kept[r] && worst_pt[r] < kept[r + 1])
                add_insert(worst_pt[r]);
        }
        struct InsertScore {
            double score;
            std::size_t q;
        };
        std::vector<InsertScore> best_joint, best_wa_only, best_rng_only;
        for (std::size_t r = 0; r < segs; ++r) {
            const std::size_t a = kept[r], c = kept[r + 1];
            if (c <= a + 1)
                continue;
            const auto [lo, hi] = probes_between(a, c);
            const double base_wa = seg_wa_score[r] * g.m_probe_weight;
            auto eval_wa_gain = [&](std::size_t j) {
                if (g.m_probe_weight <= 0.0 || lo == hi)
                    return 0.0;
                const double tj = static_cast<double>(t.t[j]);
                double s = 0.0;
                for (std::size_t p = lo; p < hi; ++p)
                    s += g.m_probe_t[p] < tj ? score_on(p, a, j) : score_on(p, j, c);
                return s * g.m_probe_weight - base_wa;
            };
            double top_joint = kMinGain, top_wa = kMinGain, top_rng = kMinGain;
            std::size_t q_joint = n, q_wa = n, q_rng = n;
            const std::size_t span = c - a;
            if (span <= 512 || (hi - lo) <= 16) {
                for (std::size_t j = a + 1; j < c; ++j) {
                    const double rg = eval_rng(j);
                    const double wg = eval_wa_gain(j);
                    if (rg > top_rng) {
                        top_rng = rg;
                        q_rng = j;
                    }
                    if (wg > top_wa) {
                        top_wa = wg;
                        q_wa = j;
                    }
                    if (rg + wg > top_joint) {
                        top_joint = rg + wg;
                        q_joint = j;
                    }
                }
            } else {
                const std::size_t step = std::max<std::size_t>(1, span / 256);
                std::size_t coarse_j = n;
                for (std::size_t j = a + 1; j < c; ++j) {
                    const double rg = eval_rng(j);
                    if (rg > top_rng) {
                        top_rng = rg;
                        q_rng = j;
                    }
                    if ((j - a) % step == 0) {
                        const double wg = eval_wa_gain(j);
                        if (wg > top_wa) {
                            top_wa = wg;
                            q_wa = j;
                        }
                        if (rg + wg > top_joint) {
                            top_joint = rg + wg;
                            coarse_j = j;
                            q_joint = j;
                        }
                    }
                }
                if (coarse_j < n) {
                    const std::size_t win_lo = coarse_j > a + step ? coarse_j - step : a + 1;
                    const std::size_t win_hi = std::min(c - 1, coarse_j + step);
                    for (std::size_t j = win_lo; j <= win_hi; ++j) {
                        const double rg = eval_rng(j);
                        const double wg = eval_wa_gain(j);
                        if (wg > top_wa) {
                            top_wa = wg;
                            q_wa = j;
                        }
                        if (rg + wg > top_joint) {
                            top_joint = rg + wg;
                            q_joint = j;
                        }
                    }
                }
                if (q_rng < n) {
                    const double wg = eval_wa_gain(q_rng);
                    if (top_rng + wg > top_joint) {
                        top_joint = top_rng + wg;
                        q_joint = q_rng;
                    }
                }
            }
            if (q_joint < n)
                best_joint.push_back({top_joint, q_joint});
            if (q_wa < n)
                best_wa_only.push_back({top_wa, q_wa});
            if (q_rng < n)
                best_rng_only.push_back({top_rng, q_rng});
        }
        auto sort_ins = [](auto &v) {
            std::ranges::sort(v, [](const InsertScore &a, const InsertScore &c) {
                return a.score > c.score || (a.score == c.score && a.q < c.q);
            });
        };
        sort_ins(best_joint);
        sort_ins(best_wa_only);
        sort_ins(best_rng_only);
        for (std::size_t k = 0; k < std::min<std::size_t>(2, best_joint.size()); ++k)
            add_insert(best_joint[k].q);
        if (!best_wa_only.empty())
            add_insert(best_wa_only[0].q);
        if (!best_rng_only.empty())
            add_insert(best_rng_only[0].q);
        return add_set;
    }
};

std::vector<std::vector<std::size_t>>
Greedy::refine_candidates(std::span<const std::size_t> kept, std::span<const double> range_weights,
                          double range_scale, std::span<const double> segments) const {
    const std::size_t n = m_trajectory.size();
    const std::size_t b = kept.size();
    if (b < 3 || b >= n)
        return {};

    LocalContext ctx(*this, kept, range_weights, range_scale, segments);
    std::vector<std::vector<std::size_t>> out;
    out.reserve(40);
    auto emit = [&](std::vector<std::size_t> cand) {
        std::ranges::sort(cand);
        if (std::ranges::equal(cand, kept))
            return;
        if (std::ranges::find(out, cand) == out.end())
            out.push_back(std::move(cand));
    };

    // 1. Knot shifts: slide interior knot kept[m] inside (kept[m-1], kept[m+1])
    struct ShiftProposal {
        double gain;
        std::size_t m;
        std::size_t replacement;
    };
    std::vector<ShiftProposal> shifts;
    shifts.reserve(b - 2);
    for (std::size_t m = 1; m + 1 < b; ++m) {
        const std::size_t a = kept[m - 1], cur = kept[m], c = kept[m + 1];
        if (c <= a + 2)
            continue;
        for (auto e = m_offsets[cur]; e < m_offsets[cur + 1]; ++e)
            --ctx.box_hits[m_point_boxes[e]];
        const auto [lo, hi] = ctx.probes_between(a, c);
        auto eval_wa = [&](std::size_t j) {
            if (m_probe_weight <= 0.0 || lo == hi)
                return 0.0;
            const double tj = static_cast<double>(m_trajectory.t[j]);
            double s = 0.0;
            for (std::size_t p = lo; p < hi; ++p)
                s += m_probe_t[p] < tj ? ctx.score_on(p, a, j) : ctx.score_on(p, j, c);
            return s * m_probe_weight;
        };
        const double cur_score =
            ctx.eval_rng(cur) + (ctx.seg_wa_score[m - 1] + ctx.seg_wa_score[m]) * m_probe_weight;
        double best_score = cur_score;
        std::size_t best_j = cur;
        const std::size_t span = c - a;
        if (span <= 512 || (hi - lo) <= 16) {
            for (std::size_t j = a + 1; j < c; ++j) {
                const double sc = ctx.eval_rng(j) + eval_wa(j);
                if (sc > best_score) {
                    best_score = sc;
                    best_j = j;
                }
            }
        } else {
            const std::size_t step = std::max<std::size_t>(1, span / 256);
            std::size_t coarse_j = cur;
            double best_rng_val = -1.0;
            std::size_t best_rng_j = cur;
            for (std::size_t j = a + 1; j < c; ++j) {
                const double rg = ctx.eval_rng(j);
                if (rg > best_rng_val) {
                    best_rng_val = rg;
                    best_rng_j = j;
                }
                if ((j - a) % step == 0) {
                    const double sc = rg + eval_wa(j);
                    if (sc > best_score) {
                        best_score = sc;
                        coarse_j = j;
                    }
                }
            }
            best_j = coarse_j;
            const std::size_t win_lo = coarse_j > a + step ? coarse_j - step : a + 1;
            const std::size_t win_hi = std::min(c - 1, coarse_j + step);
            for (std::size_t j = win_lo; j <= win_hi; ++j) {
                const double sc = ctx.eval_rng(j) + eval_wa(j);
                if (sc > best_score) {
                    best_score = sc;
                    best_j = j;
                }
            }
            if (best_rng_j != cur) {
                const double sc = best_rng_val + eval_wa(best_rng_j);
                if (sc > best_score) {
                    best_score = sc;
                    best_j = best_rng_j;
                }
            }
        }
        for (auto e = m_offsets[cur]; e < m_offsets[cur + 1]; ++e)
            ++ctx.box_hits[m_point_boxes[e]];
        if (best_j != cur && best_score > cur_score + kMinGain)
            shifts.push_back({best_score - cur_score, m, best_j});
    }
    std::ranges::sort(shifts, [](const ShiftProposal &a, const ShiftProposal &c) {
        return a.gain > c.gain || (a.gain == c.gain && a.m < c.m);
    });
    for (std::size_t k = 0; k < std::min<std::size_t>(8, shifts.size()); ++k) {
        std::vector<std::size_t> cand(kept.begin(), kept.end());
        cand[shifts[k].m] = shifts[k].replacement;
        emit(std::move(cand));
    }
    // Simultaneous red-black (even/odd) multi-knot shifts: non-adjacent knots sit on disjoint
    // intervals (kept[m-1], kept[m+1]), so shifting them together preserves order without
    // interfering on Where-At segments. Also emit a Fréchet-safe variant that never raises max dev
    const double cur_max_err = *std::ranges::max_element(ctx.seg_err);
    for (const std::size_t parity : {0uz, 1uz}) {
        std::vector<std::size_t> all_cand(kept.begin(), kept.end());
        std::vector<std::size_t> safe_cand(kept.begin(), kept.end());
        std::size_t all_cnt = 0, safe_cnt = 0;
        for (const auto &sh : shifts) {
            if ((sh.m & 1uz) != parity)
                continue;
            all_cand[sh.m] = sh.replacement;
            ++all_cnt;
            const std::size_t a = kept[sh.m - 1], c = kept[sh.m + 1];
            if (std::max(ctx.seg_deviation(a, sh.replacement).first,
                         ctx.seg_deviation(sh.replacement, c).first) <= cur_max_err) {
                safe_cand[sh.m] = sh.replacement;
                ++safe_cnt;
            }
        }
        if (all_cnt >= 2)
            emit(std::move(all_cand));
        if (safe_cnt >= 2)
            emit(std::move(safe_cand));
    }

    // Fréchet-oriented knot shifts on the knots bordering the worst 2 Fréchet segments
    for (std::size_t k = 0; k < std::min<std::size_t>(2, ctx.seg_order.size()); ++k) {
        const std::size_t r = ctx.seg_order[k];
        if (ctx.seg_err[r] <= 0.0)
            break;
        for (const std::size_t m : {r, r + 1}) {
            if (m == 0 || m + 1 >= b)
                continue;
            const std::size_t a = kept[m - 1], c = kept[m + 1];
            if (c <= a + 2)
                continue;
            for (const std::size_t rep : {ctx.rdp_peak(a, c), ctx.arc_mid(a, c)}) {
                if (rep != kept[m] && rep > a && rep < c) {
                    std::vector<std::size_t> cand(kept.begin(), kept.end());
                    cand[m] = rep;
                    emit(std::move(cand));
                }
            }
        }
    }

    // 2. Global 1-opt swaps: pair cheapest droppable knots with top global insertion points
    const auto drop_set = ctx.drop_candidates();
    const auto add_set = ctx.add_candidates();
    for (const std::size_t d : drop_set) {
        for (const std::size_t q : add_set) {
            std::vector<std::size_t> cand(kept.begin(), kept.end());
            cand[d] = q;
            emit(std::move(cand));
        }
    }
    return out;
}

Greedy::PointTradeOffers Greedy::point_trade_candidates(std::span<const std::size_t> kept,
                                                        std::span<const double> range_weights,
                                                        double range_scale,
                                                        std::span<const double> segments) const {
    PointTradeOffers offers;
    const std::size_t n = m_trajectory.size();
    const std::size_t b = kept.size();
    if (b < 2 || n < 3)
        return offers;

    LocalContext ctx(*this, kept, range_weights, range_scale, segments);
    for (const std::size_t m : ctx.drop_candidates()) {
        std::vector<std::size_t> cand;
        cand.reserve(b - 1);
        for (std::size_t k = 0; k < b; ++k)
            if (k != m)
                cand.push_back(kept[k]);
        offers.drops.push_back(std::move(cand));
    }
    for (const std::size_t q : ctx.add_candidates()) {
        std::vector<std::size_t> cand(kept.begin(), kept.end());
        cand.push_back(q);
        std::ranges::sort(cand);
        offers.adds.push_back(std::move(cand));
    }
    return offers;
}

} // namespace teacher
