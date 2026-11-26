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
      m_max_tolerance(objective.config().max_tolerance_m),
      m_min_shape(objective.config().min_shape_m), m_max_shape(objective.config().max_shape_m),
      m_arc_shape_weight(objective.db().size() > 0 ? objective.scale(Objective::Shape) * 0.5 /
                                                         static_cast<double>(objective.db().size())
                                                   : 0.0) {
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

double Greedy::score_probe(std::size_t p, std::size_t from, std::size_t to, double near_sq,
                           double far_sq) const {
    const double t_from = static_cast<double>(m_trajectory.t[from]);
    const double t_to = static_cast<double>(m_trajectory.t[to]);
    const double fraction =
        t_to > t_from ? std::clamp((m_probe_t[p] - t_from) / (t_to - t_from), 0.0, 1.0) : 0.0;
    const double x = m_trajectory.x[from] + fraction * (m_trajectory.x[to] - m_trajectory.x[from]);
    const double y = m_trajectory.y[from] + fraction * (m_trajectory.y[to] - m_trajectory.y[from]);
    const double dx = x - m_probe_x[p], dy = y - m_probe_y[p];
    const double squared = dx * dx + dy * dy;
    if (squared >= far_sq)
        return 0.0;
    if (squared <= near_sq)
        return 1.0;
    return ramp(std::hypot(dx, dy), m_min_tolerance, m_max_tolerance);
}

std::pair<std::size_t, std::size_t> Greedy::probes_between(std::size_t from, std::size_t to) const {
    if (m_probe_t.empty())
        return {0, 0};
    const auto lo = static_cast<std::size_t>(
        std::ranges::lower_bound(m_probe_t, static_cast<double>(m_trajectory.t[from])) -
        m_probe_t.begin());
    const auto hi =
        to + 1 == m_trajectory.size()
            ? m_probe_t.size()
            : static_cast<std::size_t>(
                  std::ranges::lower_bound(m_probe_t, static_cast<double>(m_trajectory.t[to])) -
                  m_probe_t.begin());
    return {lo, hi};
}

std::vector<double> Greedy::build_arc(std::span<const double> segments) const {
    const std::size_t n = m_trajectory.size();
    std::vector<double> arc(n, 0.0);
    for (std::size_t j = 0; j + 1 < n; ++j)
        arc[j + 1] =
            arc[j] + (segments.empty() ? std::hypot(m_trajectory.x[j + 1] - m_trajectory.x[j],
                                                    m_trajectory.y[j + 1] - m_trajectory.y[j])
                                       : segments[j]);
    return arc;
}

Greedy::SegGeom Greedy::eval_seg_geom(std::span<const double> arc, std::size_t from, std::size_t to,
                                      double near_sq, double far_sq) const {
    const double length = arc[to] - arc[from];
    const double ax = m_trajectory.x[from], ay = m_trajectory.y[from];
    const double dx = m_trajectory.x[to] - ax, dy = m_trajectory.y[to] - ay;
    auto shape_at = [&](double sq) {
        if (sq <= near_sq)
            return 1.0;
        if (sq >= far_sq)
            return 0.0;
        return ramp(std::sqrt(sq), m_min_shape, m_max_shape);
    };
    SegGeom out{0.0, from, 0.0, 0.0};
    double prev_ex = 0.0, prev_ey = 0.0, prev_sq = 0.0, prev_r = 1.0;
    for (std::size_t j = from + 1; j <= to; ++j) {
        const double w = arc[j] - arc[j - 1];
        const double u = length > 0.0 ? (arc[j] - arc[from]) / length : 0.0;
        const double ex = m_trajectory.x[j] - (ax + u * dx);
        const double ey = m_trajectory.y[j] - (ay + u * dy);
        const double sq = ex * ex + ey * ey;
        if (j < to && sq > out.max_sq) {
            out.max_sq = sq;
            out.worst = j;
        }
        const double r = shape_at(sq);
        if (w > 0.0) {
            out.rms_num += w * (prev_sq + prev_ex * ex + prev_ey * ey + sq) / 3.0;
            out.arc_shape_num += w * 0.5 * (prev_r + r);
        }
        prev_ex = ex;
        prev_ey = ey;
        prev_sq = sq;
        prev_r = r;
    }
    return out;
}

std::vector<std::size_t> Greedy::run(std::size_t budget, std::span<const std::size_t> fill) const {
    auto kept = sequence(budget, fill);
    std::ranges::sort(kept);
    return kept;
}

std::vector<std::size_t> Greedy::sequence(std::size_t limit,
                                          std::span<const std::size_t> fill) const {
    const std::size_t n = m_trajectory.size();
    const std::size_t budget = std::min(limit, n);
    const double far = m_max_tolerance * m_max_tolerance * (1.0 + 1e-12);
    const double near = m_min_tolerance * m_min_tolerance * (1.0 - 1e-12);

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
    std::vector<double> current_scores(m_probe_t.size());
    auto refresh = [&](std::size_t from, std::size_t to) {
        const auto [lo, hi] = probes_between(from, to);
        for (std::size_t p = lo; p < hi; ++p)
            current_scores[p] = score_probe(p, from, to, near, far);
        for (std::size_t j = from + 1; j < to; ++j) {
            double gain = 0;
            for (std::size_t p = lo; p < hi; ++p) {
                const double now = current_scores[p];
                const double then = m_probe_t[p] < static_cast<double>(m_trajectory.t[j])
                                        ? score_probe(p, from, j, near, far)
                                        : score_probe(p, j, to, near, far);
                gain += then - now;
            }
            where_at_gain[j] = gain * m_probe_weight;
        }
    };

    keep(0);
    keep(n - 1);
    if (!m_probe_t.empty() && count < budget)
        refresh(0, n - 1);

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

std::vector<std::size_t> Greedy::joint_sequence(std::size_t limit, const Objective &objective,
                                                std::size_t i, std::span<const double> segments,
                                                std::span<const std::size_t> fill) const {
    const std::size_t n = m_trajectory.size();
    const std::size_t budget = std::min(limit, n);
    if (budget <= 2 || n <= 2)
        return sequence(limit, fill);

    const double far = m_max_tolerance * m_max_tolerance * (1.0 + 1e-12);
    const double near = m_min_tolerance * m_min_tolerance * (1.0 - 1e-12);
    const double shape_far = m_max_shape * m_max_shape * (1.0 + 1e-12);
    const double shape_near = m_min_shape * m_min_shape * (1.0 - 1e-12);
    const auto arc = build_arc(segments);
    const double total_len = arc[n - 1];

    struct ActiveSeg {
        std::size_t a, c;
        double max_sq, rms_num, arc_shape_num;
        std::size_t worst;
        double split_max_sq, split_rms_num, split_arc_shape_num;
    };
    auto make_seg = [&](std::size_t a, std::size_t c) {
        const auto sg = eval_seg_geom(arc, a, c, shape_near, shape_far);
        ActiveSeg seg{a,        c,         sg.max_sq,  sg.rms_num,      sg.arc_shape_num,
                      sg.worst, sg.max_sq, sg.rms_num, sg.arc_shape_num};
        if (sg.worst > a && sg.worst < c) {
            const auto g1 = eval_seg_geom(arc, a, sg.worst, shape_near, shape_far);
            const auto g2 = eval_seg_geom(arc, sg.worst, c, shape_near, shape_far);
            seg.split_max_sq = std::max(g1.max_sq, g2.max_sq);
            seg.split_rms_num = g1.rms_num + g2.rms_num;
            seg.split_arc_shape_num = g1.arc_shape_num + g2.arc_shape_num;
        }
        return seg;
    };
    auto make_profile = [&](double max_sq, double rms_num, double arc_num) -> DeviationProfile {
        const double max_m = std::sqrt(std::max(0.0, max_sq));
        if (total_len <= 0.0)
            return {max_m, max_m, ramp(max_m, m_min_shape, m_max_shape)};
        return {max_m, std::sqrt(std::max(0.0, rms_num / total_len)), arc_num / total_len};
    };

    std::vector<char> kept(n, 0), covered(m_box_points.size(), 0);
    std::vector<double> range_gain(n, 0.0), where_at_gain(n, 0.0);
    for (std::size_t box = 0; box < m_box_points.size(); ++box)
        for (const auto k : m_box_points[box])
            range_gain[k] += m_box_weight[box];
    std::vector<std::size_t> seq;
    seq.reserve(budget);
    std::size_t count = 0;
    auto keep = [&](std::size_t j) {
        if (kept[j])
            return;
        kept[j] = 1;
        seq.push_back(j);
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
    std::vector<double> current_scores(m_probe_t.size());
    auto refresh_wa = [&](std::size_t from, std::size_t to) {
        if (m_probe_t.empty())
            return;
        const auto [lo, hi] = probes_between(from, to);
        for (std::size_t p = lo; p < hi; ++p)
            current_scores[p] = score_probe(p, from, to, near, far);
        for (std::size_t j = from + 1; j < to; ++j) {
            double gain = 0.0;
            for (std::size_t p = lo; p < hi; ++p) {
                const double now = current_scores[p];
                const double then = m_probe_t[p] < static_cast<double>(m_trajectory.t[j])
                                        ? score_probe(p, from, j, near, far)
                                        : score_probe(p, j, to, near, far);
                gain += then - now;
            }
            where_at_gain[j] = gain * m_probe_weight;
        }
    };

    keep(0);
    keep(n - 1);
    refresh_wa(0, n - 1);
    std::vector<ActiveSeg> active;
    active.reserve(budget);
    active.push_back(make_seg(0, n - 1));

    while (count < budget) {
        double max1 = -1.0, max2 = -1.0;
        std::size_t max_idx = 0;
        double sum_rms = 0.0, sum_arc = 0.0;
        for (std::size_t s = 0; s < active.size(); ++s) {
            const double e = active[s].max_sq;
            if (e > max1) {
                max2 = max1;
                max1 = e;
                max_idx = s;
            } else if (e > max2) {
                max2 = e;
            }
            sum_rms += active[s].rms_num;
            sum_arc += active[s].arc_shape_num;
        }
        const double cur_sm = objective.shape_margin(i, make_profile(max1, sum_rms, sum_arc));

        double best = kMinGain;
        std::size_t pick = n;
        std::size_t pick_seg = active.size();

        // 1. Evaluate all active segments' worst geometric split vertices in O(1) each
        for (std::size_t s = 0; s < active.size(); ++s) {
            const auto &seg = active[s];
            const std::size_t w = seg.worst;
            if (w <= seg.a || w >= seg.c || kept[w])
                continue;
            const double new_max =
                std::max(s == max_idx ? std::max(0.0, max2) : max1, seg.split_max_sq);
            const double new_rms = sum_rms - seg.rms_num + seg.split_rms_num;
            const double new_arc = sum_arc - seg.arc_shape_num + seg.split_arc_shape_num;
            const double sm_gain =
                objective.shape_margin(i, make_profile(new_max, new_rms, new_arc)) - cur_sm;
            const double total_gain = range_gain[w] + where_at_gain[w] + sm_gain;
            if (total_gain > best) {
                best = total_gain;
                pick = w;
                pick_seg = s;
            }
        }

        // 2. Evaluate the top Range+Where-At point with its exact Shape+Margin gain
        double top_rw = kMinGain;
        std::size_t rw_j = n;
        for (std::size_t j = 1; j + 1 < n; ++j)
            if (!kept[j] && range_gain[j] + where_at_gain[j] > top_rw) {
                top_rw = range_gain[j] + where_at_gain[j];
                rw_j = j;
            }
        if (rw_j < n) {
            std::size_t s = 0;
            while (s < active.size() && !(active[s].a < rw_j && rw_j < active[s].c))
                ++s;
            if (s < active.size()) {
                const auto &seg = active[s];
                const auto g1 = eval_seg_geom(arc, seg.a, rw_j, shape_near, shape_far);
                const auto g2 = eval_seg_geom(arc, rw_j, seg.c, shape_near, shape_far);
                const double new_max = std::max(s == max_idx ? std::max(0.0, max2) : max1,
                                                std::max(g1.max_sq, g2.max_sq));
                const double new_rms = sum_rms - seg.rms_num + g1.rms_num + g2.rms_num;
                const double new_arc =
                    sum_arc - seg.arc_shape_num + g1.arc_shape_num + g2.arc_shape_num;
                const double sm_gain =
                    objective.shape_margin(i, make_profile(new_max, new_rms, new_arc)) - cur_sm;
                const double total_gain = top_rw + sm_gain;
                if (total_gain > best) {
                    best = total_gain;
                    pick = rw_j;
                    pick_seg = s;
                }
            }
        }

        if (pick == n)
            break;
        keep(pick);
        if (count < budget && pick_seg < active.size()) {
            const std::size_t a = active[pick_seg].a, c = active[pick_seg].c;
            active[pick_seg] = make_seg(a, pick);
            active.insert(active.begin() + static_cast<std::ptrdiff_t>(pick_seg + 1),
                          make_seg(pick, c));
            refresh_wa(a, pick);
            refresh_wa(pick, c);
        }
    }
    for (std::size_t z = 0; z < fill.size() && count < budget; ++z)
        keep(fill[z]);
    for (std::size_t j = 1; j + 1 < n && count < budget; ++j)
        keep(j);
    return seq;
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
    double shape_far = 0.0;
    double shape_near = 0.0;
    double arc_scale = 0.0;
    std::vector<char> is_kept;
    std::vector<double> arc;
    std::vector<std::uint32_t> box_hits;
    std::vector<double> live_box_weight;
    std::vector<double> seg_err;
    std::vector<double> seg_rms;
    std::vector<double> seg_arc_shape;
    std::vector<double> seg_wa_score;
    std::vector<std::size_t> worst_pt;
    std::vector<std::size_t> seg_order;
    std::vector<std::size_t> rms_order;

    LocalContext(const Greedy &greedy, std::span<const std::size_t> kept_set,
                 std::span<const double> range_weights, double range_scale,
                 std::span<const double> segments)
        : g(greedy), kept(kept_set), n(g.m_trajectory.size()), b(kept_set.size()), segs(b - 1),
          far(g.m_max_tolerance * g.m_max_tolerance * (1.0 + 1e-12)),
          near(g.m_min_tolerance * g.m_min_tolerance * (1.0 - 1e-12)),
          shape_far(g.m_max_shape * g.m_max_shape * (1.0 + 1e-12)),
          shape_near(g.m_min_shape * g.m_min_shape * (1.0 - 1e-12)), is_kept(n, 0),
          arc(g.build_arc(segments)), box_hits(g.m_box_points.size(), 0),
          live_box_weight(g.m_box_points.size(), 0.0), seg_err(segs, 0.0), seg_rms(segs, 0.0),
          seg_arc_shape(segs, 0.0), seg_wa_score(segs, 0.0), worst_pt(segs, n), seg_order(segs),
          rms_order(segs) {
        for (const auto idx : kept)
            is_kept[idx] = 1;
        arc_scale = arc[n - 1] > 0.0 ? g.m_arc_shape_weight / arc[n - 1] : 0.0;
        for (const auto idx : kept)
            for (auto e = g.m_offsets[idx]; e < g.m_offsets[idx + 1]; ++e)
                ++box_hits[g.m_point_boxes[e]];
        for (std::size_t box = 0; box < g.m_box_points.size(); ++box)
            live_box_weight[box] =
                range_weights.empty() ? g.m_box_weight[box] : range_scale * range_weights[box];
        for (std::size_t r = 0; r < segs; ++r) {
            const auto sg = seg_geom(kept[r], kept[r + 1]);
            seg_err[r] = sg.max_sq;
            seg_rms[r] = sg.rms_num;
            seg_arc_shape[r] = sg.arc_shape_num;
            worst_pt[r] = sg.worst;
            if (g.m_probe_weight > 0.0)
                seg_wa_score[r] = seg_wa(kept[r], kept[r + 1]);
        }
        std::iota(seg_order.begin(), seg_order.end(), 0);
        std::ranges::sort(seg_order, [&](std::size_t a, std::size_t c) {
            return seg_err[a] > seg_err[c] || (seg_err[a] == seg_err[c] && a < c);
        });
        std::iota(rms_order.begin(), rms_order.end(), 0);
        std::ranges::sort(rms_order, [&](std::size_t a, std::size_t c) {
            return seg_rms[a] > seg_rms[c] || (seg_rms[a] == seg_rms[c] && a < c);
        });
    }

    double score_on(std::size_t p, std::size_t from, std::size_t to) const {
        return g.score_probe(p, from, to, near, far);
    }

    std::pair<std::size_t, std::size_t> probes_between(std::size_t from, std::size_t to) const {
        return g.probes_between(from, to);
    }

    double seg_wa(std::size_t from, std::size_t to) const {
        const auto [lo, hi] = probes_between(from, to);
        double sum = 0.0;
        for (std::size_t p = lo; p < hi; ++p)
            sum += score_on(p, from, to);
        return sum;
    }

    SegGeom seg_geom(std::size_t from, std::size_t to) const {
        return g.eval_seg_geom(arc, from, to, shape_near, shape_far);
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

    // Up to 5 cheapest droppable interior knot positions m in [1, b - 2]
    std::vector<std::size_t> drop_candidates() const {
        if (b < 3)
            return {};
        struct DropInfo {
            std::size_t m;
            double local_loss;
            double merged_err_sq;
            double rms_loss;
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
            const auto merged = seg_geom(kept[m - 1], kept[m + 1]);
            const double arc_loss =
                arc_scale * (seg_arc_shape[m - 1] + seg_arc_shape[m] - merged.arc_shape_num);
            const double rms_loss = merged.rms_num - (seg_rms[m - 1] + seg_rms[m]);
            drops.push_back({m, rng_loss + wa_loss + arc_loss, merged.max_sq, rms_loss});
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
        auto by_rms = drops;
        std::ranges::sort(by_rms, [](const DropInfo &a, const DropInfo &c) {
            return a.rms_loss < c.rms_loss || (a.rms_loss == c.rms_loss && a.m < c.m);
        });
        if (!by_rms.empty())
            add_drop(by_rms[0].m);
        return drop_set;
    }

    // Up to 8 top global unkept insertion points (Fréchet max/RMS + Range + Where-At)
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
        for (std::size_t k = 0; k < std::min<std::size_t>(2, rms_order.size()); ++k) {
            const std::size_t r = rms_order[k];
            if (seg_rms[r] > 0.0 && worst_pt[r] > kept[r] && worst_pt[r] < kept[r + 1])
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
    // intervals (kept[m-1], kept[m+1]). The safe variant preserves both max and local RMS error
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
            const auto g1 = ctx.seg_geom(a, sh.replacement);
            const auto g2 = ctx.seg_geom(sh.replacement, c);
            if (std::max(g1.max_sq, g2.max_sq) <= cur_max_err &&
                g1.rms_num + g2.rms_num <= ctx.seg_rms[sh.m - 1] + ctx.seg_rms[sh.m]) {
                safe_cand[sh.m] = sh.replacement;
                ++safe_cnt;
            }
        }
        if (all_cnt >= 2)
            emit(std::move(all_cand));
        if (safe_cnt >= 2)
            emit(std::move(safe_cand));
    }

    // Fréchet-oriented knot shifts on the knots bordering the worst 2 max-error and RMS segments
    std::vector<std::size_t> geom_segs;
    for (const auto &order_vec : {ctx.seg_order, ctx.rms_order})
        for (std::size_t k = 0; k < std::min<std::size_t>(2, order_vec.size()); ++k) {
            const std::size_t r = order_vec[k];
            if (ctx.seg_err[r] > 0.0 && std::ranges::find(geom_segs, r) == geom_segs.end())
                geom_segs.push_back(r);
        }
    for (const std::size_t r : geom_segs) {
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
