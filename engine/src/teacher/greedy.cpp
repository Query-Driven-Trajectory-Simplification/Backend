#include "backend/engine/teacher/greedy.hpp"

#include <algorithm>
#include <cmath>
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

} // namespace teacher
