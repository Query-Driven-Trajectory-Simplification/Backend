#include "backend/engine/teacher/greedy.hpp"

#include <algorithm>
#include <cmath>
#include <set>

namespace teacher {

Greedy::Greedy(const Objective &objective, std::size_t i)
    : m_v(objective.db().trajectory(i)), m_min_tolerance(objective.config().min_tolerance_m),
      m_max_tolerance(objective.config().max_tolerance_m) {
    const auto &v = m_v;
    const std::size_t n = v.size();
    const auto &workloads = objective.workloads();

    const double range_scale = objective.scale(Objective::Range);
    std::vector<std::uint32_t> count(n, 0);
    for (const auto &[b, w] : objective.boxes(i)) {
        const auto &box = workloads.range.queries[b].box;
        std::vector<std::uint32_t> inside;
        // t is sorted, so only the box's time window is scanned
        for (auto k =
                 static_cast<std::size_t>(std::ranges::lower_bound(v.t, box.t_min) - v.t.begin());
             k < n && v.t[k] <= box.t_max; ++k)
            if (v.x[k] >= box.x_min && v.x[k] <= box.x_max && v.y[k] >= box.y_min &&
                v.y[k] <= box.y_max) {
                inside.push_back(static_cast<std::uint32_t>(k));
                ++count[k];
            }
        m_box_points.push_back(std::move(inside));
        m_box_weight.push_back(range_scale * w);
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
        const auto &q = workloads.where_at.queries[k];
        m_probe_t.push_back(q.t);
        m_probe_x.push_back(q.reference.x);
        m_probe_y.push_back(q.reference.y);
    }
    if (!order.empty())
        m_probe_weight = objective.scale(Objective::WhereAt) * objective.where_at_weight() /
                         static_cast<double>(order.size());
}

std::vector<std::size_t> Greedy::run(std::size_t b, std::span<const std::size_t> seed,
                                     std::span<const std::size_t> fill) const {
    auto kept = sequence(b, seed, fill);
    std::ranges::sort(kept);
    return kept;
}

std::vector<std::size_t> Greedy::sequence(std::size_t limit, std::span<const std::size_t> seed,
                                          std::span<const std::size_t> fill) const {
    const auto &v = m_v;
    const std::size_t n = v.size();
    const std::size_t b = std::min(limit, n);

    auto ramp = [&](double error) {
        return std::clamp((m_max_tolerance - error) / (m_max_tolerance - m_min_tolerance), 0.0,
                          1.0);
    };
    // Where-at score of probe p on the segment from kept point a to kept point c
    auto score_on = [&](std::size_t p, std::size_t a, std::size_t c) {
        const double ta = static_cast<double>(v.t[a]), tc = static_cast<double>(v.t[c]);
        const double f = tc > ta ? std::clamp((m_probe_t[p] - ta) / (tc - ta), 0.0, 1.0) : 0.0;
        return ramp(std::hypot(v.x[a] + f * (v.x[c] - v.x[a]) - m_probe_x[p],
                               v.y[a] + f * (v.y[c] - v.y[a]) - m_probe_y[p]));
    };
    // Probes from t_a up to t_c, the last segment including its end
    auto probes_between = [&](std::size_t a, std::size_t c) {
        const auto lo = static_cast<std::size_t>(
            std::ranges::lower_bound(m_probe_t, static_cast<double>(v.t[a])) - m_probe_t.begin());
        const auto hi = c == n - 1
                            ? m_probe_t.size()
                            : static_cast<std::size_t>(
                                  std::ranges::lower_bound(m_probe_t, static_cast<double>(v.t[c])) -
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
    auto refresh = [&](std::size_t a, std::size_t c) {
        const auto [lo, hi] = probes_between(a, c);
        for (std::size_t p = lo; p < hi; ++p)
            current_scores[p] = score_on(p, a, c);
        for (std::size_t j = a + 1; j < c; ++j) {
            double gain = 0;
            for (std::size_t p = lo; p < hi; ++p) {
                const double now = current_scores[p];
                const double then = m_probe_t[p] < static_cast<double>(v.t[j]) ? score_on(p, a, j)
                                                                               : score_on(p, j, c);
                gain += then - now;
            }
            where_at_gain[j] = gain * m_probe_weight;
        }
    };

    keep(0);
    keep(n - 1);
    for (const auto j : seed)
        if (count < b)
            keep(j);
    if (!m_probe_t.empty() && count < b)
        for (auto it = order.begin(); std::next(it) != order.end(); ++it)
            refresh(*it, *std::next(it));

    while (count < b) {
        double best = 1e-15;
        std::size_t pick = n;
        for (std::size_t j = 1; j + 1 < n; ++j)
            if (!kept[j] && range_gain[j] + where_at_gain[j] > best) {
                best = range_gain[j] + where_at_gain[j];
                pick = j;
            }
        if (pick == n)
            break;
        keep(pick);
        if (!m_probe_t.empty() && count < b) {
            const auto it = order.find(pick);
            refresh(*std::prev(it), pick);
            refresh(pick, *std::next(it));
        }
    }
    for (std::size_t z = 0; z < fill.size() && count < b; ++z)
        keep(fill[z]);
    for (std::size_t j = 1; j + 1 < n && count < b; ++j) // only if fill ran short
        keep(j);

    return sequence;
}

} // namespace teacher
