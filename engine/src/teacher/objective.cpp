#include "backend/engine/teacher/objective.hpp"

#include "backend/engine/core/interpolation.hpp"
#include "backend/engine/core/parallel.hpp"
#include "backend/engine/query/frechet_helper.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace teacher {

namespace {

// Trajectory i with only its kept points. The Fréchet and interpolation code read t, x and y
// only, so lat/lon stay empty
class SimplifiedView {
  public:
    SimplifiedView(const core::TrajectoryView &original, std::span<const std::size_t> kept) {
        m_t.resize(kept.size());
        m_x.resize(kept.size());
        m_y.resize(kept.size());
        for (std::size_t k = 0; k < kept.size(); ++k) {
            m_t[k] = original.t[kept[k]];
            m_x[k] = original.x[kept[k]];
            m_y[k] = original.y[kept[k]];
        }
        m_view = {original.trajectory_id, m_t, {}, {}, m_x, m_y, original.bbox};
    }
    const core::TrajectoryView &view() const { return m_view; }

  private:
    std::vector<std::int64_t> m_t;
    std::vector<double> m_x, m_y;
    core::TrajectoryView m_view{};
};

bool inside(const query::RangeBox &box, const core::TrajectoryView &v, std::size_t p) {
    return v.t[p] >= box.t_min && v.t[p] <= box.t_max && v.x[p] >= box.x_min &&
           v.x[p] <= box.x_max && v.y[p] >= box.y_min && v.y[p] <= box.y_max;
}

// Engine join F1 for one trajectory: 2 hits / (found + expected)
double f1_of(std::size_t hits, std::size_t found, std::size_t expected) {
    return 2.0 * static_cast<double>(hits) / static_cast<double>(found + expected);
}

} // namespace

Objective::Objective(const core::TrajectoryDb &db, const TeacherWorkloads &workloads,
                     const ObjectiveConfig &config)
    : m_db(db), m_workloads(workloads), m_config(config) {
    const std::size_t n = db.size();
    if (workloads.knn.size() != n)
        throw std::invalid_argument("objective: kNN workload does not match the database");
    if (!(config.min_tolerance_m < config.max_tolerance_m))
        throw std::invalid_argument("objective: min_tolerance_m must be below max_tolerance_m");

    // kNN: invert each query's top k
    m_knn.resize(n);
    std::size_t valid = 0;
    for (std::size_t e = 0; e < n; ++e) {
        const auto &q = workloads.knn[e];
        if (!q.valid)
            continue;
        ++valid;
        for (const auto i : q.top)
            m_knn[i].push_back({static_cast<std::uint32_t>(e), q.cutoff});
    }
    m_knn_weight =
        valid ? 1.0 / (static_cast<double>(workloads.k) * static_cast<double>(valid)) : 0.0;

    // Range: invert each box's reference. Recall per box, averaged over boxes like range F1
    m_range.resize(n);
    std::size_t scored = 0;
    for (const auto &q : workloads.range.queries)
        scored += !q.reference.empty();
    for (std::size_t b = 0; b < workloads.range.queries.size(); ++b) {
        const auto &ref = workloads.range.queries[b].reference;
        for (const auto i : ref)
            m_range[i].push_back(
                {static_cast<std::uint32_t>(b),
                 1.0 / (static_cast<double>(ref.size()) * static_cast<double>(scored))});
    }

    // Where-at: probes per trajectory, averaged per trajectory like the evaluator
    m_probes.resize(n);
    for (std::size_t k = 0; k < workloads.where_at.queries.size(); ++k)
        m_probes[workloads.where_at.queries[k].trajectory_index].push_back(
            static_cast<std::uint32_t>(k));
    const auto with_probes = std::ranges::count_if(m_probes, [](auto &p) { return !p.empty(); });
    m_where_at_weight = with_probes ? 1.0 / static_cast<double>(with_probes) : 0.0;

    // Join: seen positives, then seen negatives
    m_links.resize(n);
    m_truth.assign(n, 0);
    auto add = [&](std::uint32_t i, std::uint32_t j, bool positive) {
        const auto pair = static_cast<std::uint32_t>(m_pair_i.size());
        m_pair_i.push_back(i);
        m_pair_j.push_back(j);
        m_links[i].push_back({j, pair, positive});
        m_links[j].push_back({i, pair, positive});
        if (positive) {
            ++m_truth[i];
            ++m_truth[j];
        }
    };
    for (const auto &p : workloads.positives)
        if (p.seen)
            add(p.i, p.j, true);
    m_positive_count = m_pair_i.size();
    for (const auto &p : workloads.negatives)
        if (p.seen)
            add(p.i, p.j, false);
    m_with_truth = static_cast<std::size_t>(std::ranges::count_if(m_truth, [](auto c) { return c > 0; }));

    // Floors: endpoints only, partners included, like the J5 floor
    KeptSets endpoints(n);
    for (std::size_t i = 0; i < n; ++i)
        endpoints[i] = {0, db.trajectory(i).size() - 1};
    m_floor = metrics(endpoints);
    for (std::size_t m = 0; m < TermCount; ++m) {
        if (m_floor[m] >= 1.0)
            throw std::runtime_error("objective: a metric is already perfect with endpoints only");
        m_scale[m] = 1.0 / (static_cast<double>(TermCount) * (1.0 - m_floor[m]));
    }
    m_v0.assign(n, 0.0);
    core::parallel_for(n, [&](std::size_t i) {
        const auto t = raw_separable(i, endpoints[i]);
        for (const auto m : {Knn, Range, WhereAt})
            m_v0[i] += m_scale[m] * t[m];
    });
}

bool Objective::match(std::size_t a, std::span<const std::size_t> kept_a, std::size_t b,
                      std::span<const std::size_t> kept_b) const {
    return query::frechet_within(SimplifiedView(m_db.trajectory(a), kept_a).view(),
                                 SimplifiedView(m_db.trajectory(b), kept_b).view(),
                                 m_workloads.join_eps_m, m_workloads.frechet);
}

bool Objective::match_original(std::size_t a, std::span<const std::size_t> kept_a,
                               std::size_t b) const {
    return query::frechet_within(SimplifiedView(m_db.trajectory(a), kept_a).view(),
                                 m_db.trajectory(b), m_workloads.join_eps_m, m_workloads.frechet);
}

Objective::Terms Objective::raw_separable(std::size_t i, std::span<const std::size_t> kept) const {
    const SimplifiedView simplified(m_db.trajectory(i), kept);
    const auto &s = simplified.view();
    Terms t{};

    for (const auto &[e, cutoff] : m_knn[i])
        if (query::frechet_within(m_db.trajectory(e), s, cutoff, m_workloads.frechet))
            t[Knn] += m_knn_weight;

    for (const auto &[b, w] : m_range[i]) {
        const auto &box = m_workloads.range.queries[b].box;
        for (std::size_t k = 0; k < s.size(); ++k)
            if (inside(box, s, k)) {
                t[Range] += w;
                break;
            }
    }

    if (!m_probes[i].empty()) {
        double sum = 0;
        for (const auto k : m_probes[i]) {
            const auto &probe = m_workloads.where_at.queries[k];
            const double error = core::distance(core::position_at(s, probe.t), probe.reference);
            sum += std::clamp((m_config.max_tolerance_m - error) /
                                  (m_config.max_tolerance_m - m_config.min_tolerance_m),
                              0.0, 1.0);
        }
        t[WhereAt] = m_where_at_weight * sum / static_cast<double>(m_probes[i].size());
    }
    return t;
}

double Objective::separable(std::size_t i, std::span<const std::size_t> kept) const {
    const auto t = raw_separable(i, kept);
    return m_scale[Knn] * t[Knn] + m_scale[Range] * t[Range] + m_scale[WhereAt] * t[WhereAt] -
           m_v0[i];
}

double Objective::independent_value(std::size_t i, std::span<const std::size_t> kept,
                                    const KeptSets *partners) const {
    double v = separable(i, kept);
    std::size_t hits = 0, found = 0;
    for (const auto &link : m_links[i]) {
        const bool matched = partners ? match(i, kept, link.other, (*partners)[link.other])
                                      : match_original(i, kept, link.other);
        found += matched;
        hits += matched && link.positive;
    }
    if (found + m_truth[i] != 0 && m_with_truth != 0)
        v += m_scale[Join] * f1_of(hits, found, m_truth[i]) / static_cast<double>(m_with_truth);
    return v;
}

double Objective::join_metric(const KeptSets &kept) const {
    std::vector<char> matched(m_pair_i.size());
    core::parallel_for(m_pair_i.size(), [&](std::size_t p) {
        matched[p] = match(m_pair_i[p], kept[m_pair_i[p]], m_pair_j[p], kept[m_pair_j[p]]);
    });
    std::vector<std::size_t> hits(m_db.size(), 0), found(m_db.size(), 0);
    for (std::size_t p = 0; p < m_pair_i.size(); ++p) {
        if (!matched[p])
            continue;
        const bool positive = p < m_positive_count;
        for (const auto t : {m_pair_i[p], m_pair_j[p]}) {
            ++found[t];
            hits[t] += positive;
        }
    }
    double sum = 0;
    std::size_t counted = 0;
    for (std::size_t t = 0; t < m_db.size(); ++t)
        if (found[t] + m_truth[t] != 0) {
            sum += f1_of(hits[t], found[t], m_truth[t]);
            ++counted;
        }
    return counted ? sum / static_cast<double>(counted) : 0.0;
}

Objective::Terms Objective::metrics(const KeptSets &kept) const {
    if (kept.size() != m_db.size())
        throw std::invalid_argument("objective: one kept set per trajectory needed");
    std::vector<Terms> per(kept.size());
    core::parallel_for(kept.size(), [&](std::size_t i) { per[i] = raw_separable(i, kept[i]); });
    Terms total{};
    for (const auto &t : per)
        for (const auto m : {Knn, Range, WhereAt})
            total[m] += t[m];
    total[Join] = join_metric(kept);
    return total;
}

double Objective::total(const KeptSets &kept) const {
    const auto m = metrics(kept);
    double j = 0;
    for (std::size_t k = 0; k < TermCount; ++k)
        j += m_scale[k] * (m[k] - m_floor[k]);
    return j;
}

JoinState::JoinState(const Objective &objective, const KeptSets &state)
    : m_objective(objective) {
    const std::size_t n = state.size();
    m_matched.assign(objective.pair_count(), 0);
    m_hits.assign(n, 0);
    m_found.assign(n, 0);
    // Each pair once, from its smaller side
    std::vector<std::vector<std::pair<std::uint32_t, char>>> outcome(n);
    core::parallel_for(n, [&](std::size_t i) {
        for (const auto &link : objective.links(i))
            if (i < link.other)
                outcome[i].push_back(
                    {link.pair, objective.match(i, state[i], link.other, state[link.other])});
    });
    for (std::size_t i = 0; i < n; ++i)
        for (const auto &[pair, matched] : outcome[i])
            m_matched[pair] = matched;
    for (std::size_t i = 0; i < n; ++i)
        for (const auto &link : objective.links(i))
            if (m_matched[link.pair]) {
                ++m_found[i];
                m_hits[i] += link.positive;
            }
    for (std::size_t t = 0; t < n; ++t)
        if (counted(t, m_found[t])) {
            m_sum += f1(t, m_hits[t], m_found[t]);
            ++m_counted;
        }
}

bool JoinState::counted(std::size_t t, std::size_t found) const {
    return found + m_objective.truth(t) != 0;
}

double JoinState::f1(std::size_t t, std::size_t hits, std::size_t found) const {
    return f1_of(hits, found, m_objective.truth(t));
}

double JoinState::metric() const {
    return m_counted ? m_sum / static_cast<double>(m_counted) : 0.0;
}

double JoinState::delta(std::size_t i, std::span<const std::size_t> kept,
                        const KeptSets &state) const {
    // New outcomes of i's pairs, and the hits/found changes they cause on each side
    std::unordered_map<std::size_t, std::pair<long, long>> change; // t -> (hits, found)
    for (const auto &link : m_objective.links(i)) {
        const bool now = m_matched[link.pair];
        const bool then = m_objective.match(i, kept, link.other, state[link.other]);
        if (now == then)
            continue;
        const long d = then ? 1 : -1;
        for (const auto t : {i, static_cast<std::size_t>(link.other)}) {
            auto &[h, f] = change[t];
            f += d;
            h += link.positive ? d : 0;
        }
    }
    if (change.empty())
        return 0.0;
    double sum = m_sum;
    long n_counted = static_cast<long>(m_counted);
    for (const auto &[t, c] : change) {
        const auto old_found = m_found[t], old_hits = m_hits[t];
        const auto new_found = static_cast<std::size_t>(static_cast<long>(old_found) + c.second);
        const auto new_hits = static_cast<std::size_t>(static_cast<long>(old_hits) + c.first);
        if (counted(t, old_found)) {
            sum -= f1(t, old_hits, old_found);
            --n_counted;
        }
        if (counted(t, new_found)) {
            sum += f1(t, new_hits, new_found);
            ++n_counted;
        }
    }
    const double after = n_counted ? sum / static_cast<double>(n_counted) : 0.0;
    return m_objective.scale(Objective::Join) * (after - metric());
}

void JoinState::apply(std::size_t i, const KeptSets &state) {
    for (const auto &link : m_objective.links(i)) {
        const bool now = m_matched[link.pair];
        const bool then = m_objective.match(i, state[i], link.other, state[link.other]);
        if (now == then)
            continue;
        m_matched[link.pair] = then;
        const long d = then ? 1 : -1;
        for (const auto t : {i, static_cast<std::size_t>(link.other)}) {
            if (counted(t, m_found[t])) {
                m_sum -= f1(t, m_hits[t], m_found[t]);
                --m_counted;
            }
            m_found[t] = static_cast<std::size_t>(static_cast<long>(m_found[t]) + d);
            if (link.positive)
                m_hits[t] = static_cast<std::size_t>(static_cast<long>(m_hits[t]) + d);
            if (counted(t, m_found[t])) {
                m_sum += f1(t, m_hits[t], m_found[t]);
                ++m_counted;
            }
        }
    }
}

} // namespace teacher
