#include "backend/engine/teacher/objective.hpp"

#include "backend/engine/core/interpolation.hpp"
#include "backend/engine/core/parallel.hpp"
#include "backend/engine/core/simplification.hpp"
#include "backend/engine/query/frechet_helper.hpp"
#include "backend/engine/query/knn.hpp"
#include "backend/engine/query/range_query.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace teacher {

PreparedTrajectory::PreparedTrajectory(const core::TrajectoryView &original,
                                       std::span<const std::size_t> kept)
    : m_id(original.trajectory_id) {
    assign(original, kept);
}

void PreparedTrajectory::assign(const core::TrajectoryView &original,
                                std::span<const std::size_t> kept) {
    m_id = original.trajectory_id;
    m_t.resize(kept.size());
    m_x.resize(kept.size());
    m_y.resize(kept.size());
    constexpr double inf = std::numeric_limits<double>::infinity();
    m_box = {inf, -inf, inf, -inf};
    for (std::size_t k = 0; k < kept.size(); ++k) {
        m_t[k] = original.t[kept[k]];
        m_x[k] = original.x[kept[k]];
        m_y[k] = original.y[kept[k]];
        m_box.x_min = std::min(m_box.x_min, m_x[k]);
        m_box.x_max = std::max(m_box.x_max, m_x[k]);
        m_box.y_min = std::min(m_box.y_min, m_y[k]);
        m_box.y_max = std::max(m_box.y_max, m_y[k]);
    }
}

PreparedTrajectories::PreparedTrajectories(const core::TrajectoryDb &db, const KeptSets &kept) {
    if (kept.size() != db.size())
        throw std::invalid_argument("prepared trajectories: one kept set per trajectory needed");
    m_trajectories.reserve(kept.size());
    for (std::size_t i = 0; i < kept.size(); ++i)
        m_trajectories.emplace_back(db.trajectory(i), kept[i]);
}

std::vector<core::TrajectoryView> PreparedTrajectories::views() const {
    std::vector<core::TrajectoryView> views;
    views.reserve(m_trajectories.size());
    for (const auto &trajectory : m_trajectories)
        views.push_back(trajectory.view());
    return views;
}

namespace {

double range_f1(std::size_t found, std::size_t expected) {
    return 2.0 * static_cast<double>(found) / static_cast<double>(expected + found);
}

double range_marginal(std::size_t others, std::size_t expected, std::size_t scored) {
    // F1(others + 1) - F1(others), avoiding subtraction of nearly equal values
    const double denominator = static_cast<double>(expected + others);
    return 2.0 * static_cast<double>(expected) /
           (denominator * (denominator + 1.0) * static_cast<double>(scored));
}

constexpr std::size_t kMaxJoinTerms = 8;
// Relative margin for certificates, far above the decision predicate's 1e-12 rounding slack
constexpr double kCertificateMargin = 1e-9;

// Upper bound on the continuous Fréchet distance between original and a simplification keeping
// a subsequence of its points: each kept segment walks its original stretch at the same share
// of arc length. Along each original edge the gap is convex, so its maximum is at a vertex
double simplification_bound(const core::TrajectoryView &original,
                            const core::TrajectoryView &kept) {
    const std::size_t n = original.size();
    if (kept.size() < 2 || n < 2)
        return std::numeric_limits<double>::infinity();
    double bound = 0.0;
    std::size_t at = 0;
    for (std::size_t k = 0; k + 1 < kept.size(); ++k) {
        // Kept points are copies of original points, in order
        while (at < n && !(original.t[at] == kept.t[k] && original.x[at] == kept.x[k] &&
                           original.y[at] == kept.y[k]))
            ++at;
        std::size_t end = at + 1;
        while (end < n && !(original.t[end] == kept.t[k + 1] && original.x[end] == kept.x[k + 1] &&
                            original.y[end] == kept.y[k + 1]))
            ++end;
        if (end >= n)
            return std::numeric_limits<double>::infinity();
        double length = 0.0;
        for (std::size_t j = at; j < end; ++j)
            length += std::hypot(original.x[j + 1] - original.x[j], original.y[j + 1] - original.y[j]);
        double walked = 0.0;
        for (std::size_t j = at; j <= end; ++j) {
            if (j > at)
                walked += std::hypot(original.x[j] - original.x[j - 1], original.y[j] - original.y[j - 1]);
            const double u = length > 0 ? walked / length : 0.0;
            const double x = kept.x[k] + u * (kept.x[k + 1] - kept.x[k]);
            const double y = kept.y[k] + u * (kept.y[k + 1] - kept.y[k]);
            bound = std::max(bound, std::hypot(original.x[j] - x, original.y[j] - y));
        }
        at = end;
    }
    return bound;
}
// Cached kNN distances are upper bounds on a fixed millimetre grid.
constexpr double kKnnDistanceError = query::kKnnResolutionM;

double f1_of(std::size_t hits, std::size_t found, std::size_t expected) {
    return 2.0 * static_cast<double>(hits) / static_cast<double>(found + expected);
}

// The join metric from each term's sum of F1 over its counted slots
double mean_f1(const Objective &objective, std::span<const double> sum,
               std::span<const long> counted) {
    std::array<double, kMaxJoinTerms> f1s{};
    for (std::size_t k = 0; k < sum.size(); ++k)
        f1s[k] = counted[k] ? sum[k] / static_cast<double>(counted[k]) : 0.0;
    return objective.join_value(std::span(f1s).first(sum.size()));
}

} // namespace

double check_term_weights(const TermWeights &weights) {
    double sum = 0;
    for (const double weight : weights) {
        if (!std::isfinite(weight) || weight < 0)
            throw std::invalid_argument("term weights must be finite and not negative");
        sum += weight;
    }
    if (sum <= 0)
        throw std::invalid_argument("at least one term weight must be positive");
    return sum;
}

Objective::Objective(const core::TrajectoryDb &db, const TeacherWorkloads &workloads,
                     const ObjectiveConfig &config)
    : m_db(db), m_workloads(workloads), m_config(config) {
    const std::size_t n = db.size();
    if (workloads.knn.size() != n)
        throw std::invalid_argument("objective: kNN workload does not match the database");
    if (!(config.min_tolerance_m < config.max_tolerance_m))
        throw std::invalid_argument("objective: min_tolerance_m must be below max_tolerance_m");
    const double weight_sum = check_term_weights(config.term_weights);
    if (config.compressed_knn_query && !config.ranking_knn)
        throw std::invalid_argument("objective: compressed kNN queries need the ranking kNN term");

    // kNN: invert each query's top k
    m_knn.resize(n);
    std::size_t valid = 0;
    for (std::size_t e = 0; e < n; ++e) {
        const auto &query = workloads.knn[e];
        if (!query.valid)
            continue;
        ++valid;
        for (const auto i : query.top)
            m_knn[i].push_back({static_cast<std::uint32_t>(e), query.cutoff});
    }
    m_knn_weight =
        valid ? 1.0 / (static_cast<double>(workloads.k) * static_cast<double>(valid)) : 0.0;

    // Range: invert each box's reference
    m_range.resize(n);
    std::size_t scored = 0;
    for (const auto &query : workloads.range.queries)
        scored += !query.reference.empty();
    for (std::size_t b = 0; b < workloads.range.queries.size(); ++b) {
        const auto &ref = workloads.range.queries[b].reference;
        for (const auto i : ref)
            m_range[i].push_back({static_cast<std::uint32_t>(b),
                                  range_marginal(ref.size() - 1, ref.size(), scored)});
    }

    // Where-at: probes per trajectory
    m_probes.resize(n);
    for (std::size_t k = 0; k < workloads.where_at.queries.size(); ++k)
        m_probes[workloads.where_at.queries[k].trajectory_index].push_back(
            static_cast<std::uint32_t>(k));
    const auto with_probes =
        std::ranges::count_if(m_probes, [](auto &probes) { return !probes.empty(); });
    m_where_at_weight = with_probes ? 1.0 / static_cast<double>(with_probes) : 0.0;

    // Join terms: join_eps_m, the extra thresholds, then the held-out form of the first. Pairs are
    // stored positives first, then negatives
    struct TermPairs {
        const std::vector<JoinPair> *positives, *negatives;
    };
    const bool held_out = config.join_mode == query::JoinMode::HeldOut;
    std::vector<TermPairs> terms{{&workloads.positives, &workloads.negatives}};
    m_eps = {workloads.join_eps_m};
    m_term_held_out = {held_out};
    for (const auto &extra : workloads.extra_joins) {
        terms.push_back({&extra.positives, &extra.negatives});
        m_eps.push_back(extra.eps_m);
        m_term_held_out.push_back(held_out);
    }
    if (config.held_out_term && !held_out) {
        terms.push_back(terms.front());
        m_eps.push_back(workloads.join_eps_m);
        m_term_held_out.push_back(true);
    }
    if (terms.size() > kMaxJoinTerms)
        throw std::invalid_argument("objective: too many join terms");
    m_links.resize(n);
    m_truth.assign(n * terms.size(), 0);
    auto add = [&](const JoinPair &join_pair, bool positive, std::size_t term) {
        const auto pair = static_cast<std::uint32_t>(m_pair_i.size());
        const auto k = static_cast<std::uint8_t>(term);
        m_pair_i.push_back(join_pair.i);
        m_pair_j.push_back(join_pair.j);
        m_pair_term.push_back(k);
        m_links[join_pair.i].push_back({join_pair.j, pair, positive, k});
        m_links[join_pair.j].push_back({join_pair.i, pair, positive, k});
        if (positive) {
            ++m_truth[slot(join_pair.i, term)];
            ++m_truth[slot(join_pair.j, term)];
        }
    };
    for (std::size_t k = 0; k < terms.size(); ++k)
        for (const auto &join_pair : *terms[k].positives)
            add(join_pair, true, k);
    m_positive_count = m_pair_i.size();
    for (std::size_t k = 0; k < terms.size(); ++k)
        for (const auto &join_pair : *terms[k].negatives)
            add(join_pair, false, k);
    m_with_truth.assign(terms.size(), 0);
    for (std::size_t t = 0; t < m_truth.size(); ++t)
        m_with_truth[t / n] += m_truth[t] > 0;

    // Held-out positives: the originals' discrete Fréchet distance bounds the continuous one
    m_pair_upper.assign(m_pair_i.size(), std::numeric_limits<double>::infinity());
    core::parallel_for(m_positive_count, [&](std::size_t p) {
        const auto k = m_pair_term[p];
        if (!m_term_held_out[k] || m_workloads.frechet != query::FrechetVariant::Continuous)
            return;
        std::vector<double> prev, curr;
        const double cutoff = m_eps[k] * m_eps[k];
        const double squared = query::get_squared_frechet_distance(
            db.trajectory(m_pair_i[p]), db.trajectory(m_pair_j[p]), cutoff, prev, curr);
        if (squared < cutoff)
            m_pair_upper[p] = std::sqrt(squared);
    });

    // Floors: endpoints only, partners included, like the J4 floor. Each join term has its own,
    // so the join metric is 0 there. Terms without true pairs are left out, as J4 skips NaNs
    const auto endpoints = core::endpoints_only(db);
    m_join_floor = join_f1s(PreparedTrajectories(db, endpoints).views());
    const auto active = std::ranges::count_if(m_with_truth, [](auto count) { return count > 0; });
    m_join_weight.assign(terms.size(), 0.0);
    for (std::size_t k = 0; k < terms.size(); ++k) {
        if (m_with_truth[k] == 0)
            continue;
        if (m_join_floor[k] >= 1.0)
            throw std::runtime_error(
                "objective: a join term is already perfect with endpoints only");
        m_join_weight[k] = 1.0 / (static_cast<double>(active) * (1.0 - m_join_floor[k]));
    }
    m_floor = metrics(endpoints);
    for (std::size_t m = 0; m < TermCount; ++m) {
        if (m_floor[m] >= 1.0)
            throw std::runtime_error("objective: a metric is already perfect with endpoints only");
        m_scale[m] = config.term_weights[m] / (weight_sum * (1.0 - m_floor[m]));
    }
    m_endpoint_separable.assign(n, 0.0);
    core::parallel_for(n, [&](std::size_t i) {
        const auto endpoint_terms = raw_separable(
            i, PreparedTrajectory(db.trajectory(i), endpoints[i]).view(), !config.ranking_knn);
        for (const auto m : {Knn, WhereAt})
            m_endpoint_separable[i] += m_scale[m] * endpoint_terms[m];
    });
}

bool Objective::held_out_match(std::size_t i, const JoinLink &link,
                               const core::TrajectoryView &candidate, double &own_error) const {
    // d(original partner, candidate) <= d(originals) + d(original i, candidate)
    if (std::isfinite(m_pair_upper[link.pair])) {
        if (own_error < 0)
            own_error = simplification_bound(m_db.trajectory(i), candidate);
        if (m_pair_upper[link.pair] + own_error <= m_eps[link.term] * (1.0 - kCertificateMargin))
            return true;
    }
    return match(m_db.trajectory(link.other), candidate, link.term);
}

bool Objective::match(const core::TrajectoryView &a, const core::TrajectoryView &b,
                      std::size_t term) const {
    return query::frechet_within(a, b, m_eps[term], m_workloads.frechet);
}

Objective::Terms Objective::raw_separable(std::size_t i, std::span<const std::size_t> kept) const {
    return raw_separable(i, PreparedTrajectory(m_db.trajectory(i), kept).view());
}

Objective::Terms Objective::raw_separable(std::size_t i, const core::TrajectoryView &candidate,
                                          bool knn) const {
    Terms terms{};

    if (knn) {
        for (const auto &membership : m_knn[i])
            if (query::frechet_within(m_db.trajectory(membership.query), candidate,
                                      membership.cutoff, m_workloads.frechet))
                terms[Knn] += m_knn_weight;
    }

    if (!m_probes[i].empty()) {
        double sum = 0;
        for (const auto k : m_probes[i]) {
            const auto &probe = m_workloads.where_at.queries[k];
            const double error =
                core::distance(core::position_at(candidate, probe.t), probe.reference);
            sum += std::clamp((m_config.max_tolerance_m - error) /
                                  (m_config.max_tolerance_m - m_config.min_tolerance_m),
                              0.0, 1.0);
        }
        terms[WhereAt] = m_where_at_weight * sum / static_cast<double>(m_probes[i].size());
    }
    return terms;
}

double Objective::separable(std::size_t i, std::span<const std::size_t> kept) const {
    return separable(i, PreparedTrajectory(m_db.trajectory(i), kept).view());
}

double Objective::separable(std::size_t i, const core::TrajectoryView &candidate) const {
    const auto t = raw_separable(i, candidate, !m_config.ranking_knn);
    return m_scale[Knn] * t[Knn] + m_scale[WhereAt] * t[WhereAt] - m_endpoint_separable[i];
}

std::vector<double> Objective::range_weights(std::size_t i,
                                             std::span<const core::TrajectoryView> partners) const {
    if (m_range[i].empty())
        return {};
    const auto scored = std::ranges::count_if(m_workloads.range.queries,
                                              [](const auto &q) { return !q.reference.empty(); });
    std::vector<double> weights;
    weights.reserve(m_range[i].size());
    for (const auto &membership : m_range[i]) {
        const auto &q = m_workloads.range.queries[membership.box];
        std::size_t others = 0;
        for (const auto j : q.reference)
            if (j != i && (partners.empty() || query::intersects(partners[j], q.box)))
                ++others;
        weights.push_back(range_marginal(others, q.reference.size(), scored));
    }
    return weights;
}

double Objective::independent_value(std::size_t i, const core::TrajectoryView &candidate,
                                    std::span<const core::TrajectoryView> partners, double beat,
                                    std::span<const double> range_weights) const {
    return independent_value(prepare_independent(i, candidate, partners, range_weights), beat);
}

Objective::IndependentScore
Objective::prepare_independent(std::size_t i, const core::TrajectoryView &candidate,
                               std::span<const core::TrajectoryView> partners,
                               std::span<const double> range_weights) const {
    IndependentScore score;
    score.trajectory = i;
    score.candidate = candidate;
    score.partners = partners;
    score.terms = raw_separable(i, candidate, false);
    const auto owned_weights =
        range_weights.empty() ? this->range_weights(i, partners) : std::vector<double>{};
    if (range_weights.empty())
        range_weights = owned_weights;
    if (range_weights.size() != m_range[i].size())
        throw std::invalid_argument("objective: one range weight per membership needed");
    for (std::size_t k = 0; k < m_range[i].size(); ++k)
        if (query::intersects(candidate, m_workloads.range.queries[m_range[i][k].box].box))
            score.terms[Range] += range_weights[k];
    std::array<std::size_t, kMaxJoinTerms> hits{}, found{};
    double own_error = -1.0;
    for (const auto &link : m_links[i]) {
        // Held out: the partner is the original query and i the database side
        const bool matched =
            m_term_held_out[link.term]
                ? held_out_match(i, link, candidate, own_error)
                : match(candidate,
                        partners.empty() ? m_db.trajectory(link.other) : partners[link.other],
                        link.term);
        found[link.term] += matched;
        hits[link.term] += matched && link.positive;
    }
    double join = 0;
    for (std::size_t k = 0; k < m_eps.size(); ++k)
        if (found[k] + m_truth[slot(i, k)] != 0 && m_with_truth[k] != 0)
            join += m_join_weight[k] * f1_of(hits[k], found[k], m_truth[slot(i, k)]) /
                    static_cast<double>(m_with_truth[k]);
    score.join = m_scale[Join] * join;
    score.bound = independent_sum(score, score.terms[Knn]) +
                  m_scale[Knn] * knn_check_weight() * static_cast<double>(knn_checks(i));
    return score;
}

double Objective::independent_sum(const IndependentScore &score, double knn) const {
    // Range is i's marginal F1 against fixed partners; the loop tracks the actual joint F1
    return m_scale[Knn] * knn + m_scale[Range] * score.terms[Range] +
           m_scale[WhereAt] * score.terms[WhereAt] - m_endpoint_separable[score.trajectory] +
           score.join;
}

double Objective::independent_value(const IndependentScore &score, double beat) const {
    const auto i = score.trajectory;
    double knn = score.terms[Knn];
    auto value = [&] { return independent_sum(score, knn); };
    // Stop once even all remaining hits cannot beat beat
    const double weight = knn_check_weight();
    const double per_hit = m_scale[Knn] * weight;
    std::size_t left = knn_checks(i);
    auto check = [&](const core::TrajectoryView &a, const core::TrajectoryView &b, double cutoff) {
        if (value() + per_hit * static_cast<double>(left) < beat - kBoundSlack)
            return false;
        --left;
        if (query::frechet_within(a, b, cutoff, m_workloads.frechet))
            knn += weight;
        return true;
    };
    if (!m_config.compressed_knn_query) {
        for (const auto &m : m_knn[i])
            if (!check(m_db.trajectory(m.query), score.candidate, m.cutoff))
                return value();
        return value();
    }
    // Compressed queries: i is found by the queries it belongs to, and finds its own neighbours
    auto partner = [&](std::size_t j) {
        return score.partners.empty() ? m_db.trajectory(j) : score.partners[j];
    };
    for (const auto &m : m_knn[i])
        if (!check(partner(m.query), score.candidate, m.cutoff))
            return value();
    const auto &own = m_workloads.knn[i];
    if (own.valid)
        for (const auto j : own.top)
            if (!check(score.candidate, partner(j), own.cutoff))
                return value();
    return value();
}

std::size_t Objective::knn_checks(std::size_t i) const {
    const auto &own = m_workloads.knn[i];
    return m_knn[i].size() + (m_config.compressed_knn_query && own.valid ? own.top.size() : 0);
}

double Objective::knn_check_weight() const {
    // Compressed queries check each (query, neighbour) hit from both sides, so each counts half
    return m_config.compressed_knn_query ? m_knn_weight / 2.0 : m_knn_weight;
}

double Objective::independent_bound(std::size_t i, const core::TrajectoryView &candidate,
                                    std::span<const core::TrajectoryView> partners) const {
    return prepare_independent(i, candidate, partners).bound;
}

double Objective::join_value(std::span<const double> f1s) const {
    double value = 0;
    for (std::size_t k = 0; k < f1s.size(); ++k)
        value += m_join_weight[k] * (f1s[k] - m_join_floor[k]);
    return value;
}

double Objective::join_metric(std::span<const core::TrajectoryView> state) const {
    return join_value(join_f1s(state));
}

std::vector<double> Objective::join_f1s(std::span<const core::TrajectoryView> state) const {
    // matched[p]: the pair's outcome, counted for i. reverse[p]: counted for j, the same outcome
    // unless held out, where each side is in turn the original query
    std::vector<char> matched(m_pair_i.size()), reverse(m_pair_i.size());
    core::parallel_for(m_pair_i.size(), [&](std::size_t p) {
        const auto i = m_pair_i[p], j = m_pair_j[p];
        const auto k = m_pair_term[p];
        if (m_term_held_out[k]) {
            matched[p] = match(m_db.trajectory(i), state[j], k);
            reverse[p] = match(m_db.trajectory(j), state[i], k);
        } else {
            matched[p] = reverse[p] = match(state[i], state[j], k);
        }
    });
    std::vector<std::size_t> hits(m_truth.size(), 0), found(m_truth.size(), 0);
    for (std::size_t p = 0; p < m_pair_i.size(); ++p) {
        const bool positive = p < m_positive_count;
        const auto slot_i = slot(m_pair_i[p], m_pair_term[p]);
        const auto slot_j = slot(m_pair_j[p], m_pair_term[p]);
        if (matched[p]) {
            ++found[slot_i];
            hits[slot_i] += positive;
        }
        if (reverse[p]) {
            ++found[slot_j];
            hits[slot_j] += positive;
        }
    }
    std::vector<double> f1s(m_eps.size(), 0.0);
    std::vector<std::size_t> counted(m_eps.size(), 0);
    for (std::size_t t = 0; t < m_truth.size(); ++t)
        if (found[t] + m_truth[t] != 0) {
            f1s[t / m_db.size()] += f1_of(hits[t], found[t], m_truth[t]);
            ++counted[t / m_db.size()];
        }
    for (std::size_t k = 0; k < f1s.size(); ++k)
        f1s[k] = counted[k] ? f1s[k] / static_cast<double>(counted[k]) : 0.0;
    return f1s;
}

Objective::Terms Objective::metrics(const KeptSets &kept) const {
    if (kept.size() != m_db.size())
        throw std::invalid_argument("objective: one kept set per trajectory needed");
    const PreparedTrajectories prepared(m_db, kept);
    const auto views = prepared.views();
    std::vector<Terms> per(kept.size());
    core::parallel_for(kept.size(), [&](std::size_t i) {
        per[i] = raw_separable(i, views[i], !m_config.ranking_knn);
    });
    Terms total{};
    for (const auto &terms : per)
        for (const auto m : {Knn, WhereAt})
            total[m] += terms[m];
    total[Range] = RangeState(*this, views).metric();
    total[Join] = join_metric(views);
    if (m_config.ranking_knn)
        total[Knn] = knn_recall(core::materialize(m_db, kept));
    return total;
}

double Objective::knn_recall(const core::TrajectoryDb &compressed) const {
    const std::size_t k = m_workloads.k;
    std::vector<std::size_t> hits(m_db.size(), 0);
    core::parallel_for(m_db.size(), [&](std::size_t e) {
        const auto &entry = m_workloads.knn[e];
        if (!entry.valid)
            return;
        const auto query =
            m_config.compressed_knn_query ? compressed.trajectory(e) : m_db.trajectory(e);
        auto found =
            query::knn(compressed, query, std::min(k + 1, m_db.size()), m_workloads.frechet);
        std::erase(found, e);
        found.resize(std::min(found.size(), k));
        for (const auto i : entry.top)
            hits[e] += std::ranges::find(found, std::size_t{i}) != found.end();
    });
    return m_knn_weight * static_cast<double>(std::accumulate(hits.begin(), hits.end(), 0ul));
}

double Objective::total(const KeptSets &kept) const {
    const auto values = metrics(kept);
    double sum = 0;
    for (std::size_t k = 0; k < TermCount; ++k)
        sum += m_scale[k] * (values[k] - m_floor[k]);
    return sum;
}

RangeState::RangeState(const Objective &objective, std::span<const core::TrajectoryView> state)
    : m_objective(objective), m_found(objective.workloads().range.queries.size(), 0),
      m_covered(objective.db().size()) {
    if (!state.empty() && state.size() != objective.db().size())
        throw std::invalid_argument("range state: one view per trajectory needed");
    core::parallel_for(objective.db().size(), [&](std::size_t i) {
        const auto trajectory = state.empty() ? objective.db().trajectory(i) : state[i];
        m_covered[i].reserve(objective.boxes(i).size());
        for (const auto &membership : objective.boxes(i)) {
            const auto &q = objective.workloads().range.queries[membership.box];
            const bool covered = state.empty() || query::intersects(trajectory, q.box);
            m_covered[i].push_back(covered);
        }
    });
    for (std::size_t i = 0; i < objective.db().size(); ++i) {
        const auto memberships = objective.boxes(i);
        for (std::size_t k = 0; k < memberships.size(); ++k)
            m_found[memberships[k].box] += m_covered[i][k];
    }
    for (std::size_t b = 0; b < m_found.size(); ++b)
        if (!objective.workloads().range.queries[b].reference.empty()) {
            ++m_scored;
            m_sum += f1(b, m_found[b]);
        }
}

double RangeState::f1(std::size_t box, std::size_t found) const {
    return range_f1(found, m_objective.workloads().range.queries[box].reference.size());
}

RangeState::Proposal RangeState::propose(std::size_t i,
                                         const core::TrajectoryView &candidate) const {
    return propose(i, coverage(i, candidate));
}

std::vector<char> RangeState::coverage(std::size_t i, const core::TrajectoryView &candidate) const {
    const auto memberships = m_objective.boxes(i);
    std::vector<char> covered;
    covered.reserve(memberships.size());
    for (const auto &membership : memberships)
        covered.push_back(
            query::intersects(candidate, m_objective.workloads().range.queries[membership.box].box));
    return covered;
}

RangeState::Proposal RangeState::propose(std::size_t i, std::vector<char> covered) const {
    Proposal proposal{i, 0.0, std::move(covered)};
    const auto memberships = m_objective.boxes(i);
    if (proposal.covered.size() != memberships.size())
        throw std::invalid_argument("range proposal: one outcome per box needed");
    double change = 0.0;
    for (std::size_t k = 0; k < memberships.size(); ++k) {
        const auto b = memberships[k].box;
        const auto next = m_found[b] - m_covered[i][k] + proposal.covered[k];
        change += f1(b, next) - f1(b, m_found[b]);
    }
    if (m_scored)
        proposal.gain =
            m_objective.scale(Objective::Range) * change / static_cast<double>(m_scored);
    return proposal;
}

void RangeState::apply(const Proposal &proposal) {
    const auto memberships = m_objective.boxes(proposal.trajectory);
    auto &covered = m_covered[proposal.trajectory];
    for (std::size_t k = 0; k < memberships.size(); ++k) {
        const auto b = memberships[k].box;
        const auto next = m_found[b] - covered[k] + proposal.covered[k];
        m_sum += f1(b, next) - f1(b, m_found[b]);
        m_found[b] = next;
        covered[k] = proposal.covered[k];
    }
}

double RangeState::metric() const { return m_scored ? m_sum / static_cast<double>(m_scored) : 0.0; }

std::vector<double> RangeState::weights(std::size_t i) const {
    std::vector<double> weights;
    const auto memberships = m_objective.boxes(i);
    weights.reserve(memberships.size());
    for (std::size_t k = 0; k < memberships.size(); ++k) {
        const auto b = memberships[k].box;
        weights.push_back(range_marginal(m_found[b] - m_covered[i][k],
                                         m_objective.workloads().range.queries[b].reference.size(),
                                         m_scored));
    }
    return weights;
}

JoinState::JoinState(const Objective &objective, std::span<const core::TrajectoryView> state)
    : m_objective(objective) {
    const std::size_t n = state.size();
    m_matched.assign(2 * objective.pair_count(), 0);
    m_hits.assign(objective.slots(), 0);
    m_found.assign(objective.slots(), 0);
    m_sum.assign(objective.join_terms(), 0.0);
    m_counted.assign(objective.join_terms(), 0);
    // Each pair once, from its smaller side. Held out: each direction from its compressed side
    std::vector<std::vector<std::pair<std::size_t, char>>> outcome(n);
    core::parallel_for(n, [&](std::size_t i) {
        double own_error = -1.0;
        for (const auto &link : objective.links(i)) {
            if (objective.held_out(link.term))
                outcome[i].push_back(
                    {slot(i, link), objective.held_out_match(i, link, state[i], own_error)});
            else if (i < link.other)
                outcome[i].push_back(
                    {slot(i, link), objective.match(state[i], state[link.other], link.term)});
        }
    });
    for (std::size_t i = 0; i < n; ++i)
        for (const auto &[at, matched] : outcome[i])
            m_matched[at] = matched;
    for (std::size_t i = 0; i < n; ++i)
        for (const auto &link : objective.links(i))
            if (m_matched[slot(i, link)]) {
                // Held out, the match is found by the query, not by the compressed side
                const std::size_t t = count(objective.held_out(link.term) ? link.other : i, link);
                ++m_found[t];
                m_hits[t] += link.positive;
            }
    for (std::size_t t = 0; t < objective.slots(); ++t)
        if (counted(t, m_found[t])) {
            m_sum[t / n] += f1(t, m_hits[t], m_found[t]);
            ++m_counted[t / n];
        }
}

bool JoinState::counted(std::size_t t, std::size_t found) const {
    return found + m_objective.truth(t) != 0;
}

double JoinState::f1(std::size_t t, std::size_t hits, std::size_t found) const {
    return f1_of(hits, found, m_objective.truth(t));
}

double JoinState::metric() const { return mean_f1(m_objective, m_sum, m_counted); }

JoinState::Proposal JoinState::propose(std::size_t i, const core::TrajectoryView &candidate,
                                       std::span<const core::TrajectoryView> state) const {
    return propose(i, outcomes(i, candidate, state));
}

std::vector<char> JoinState::outcomes(std::size_t i, const core::TrajectoryView &candidate,
                                      std::span<const core::TrajectoryView> state,
                                      std::span<const char> held_out_known) const {
    const auto links = m_objective.links(i);
    if (!held_out_known.empty() && held_out_known.size() != links.size())
        throw std::invalid_argument("join outcomes: one known outcome per link needed");
    std::vector<char> matched;
    matched.reserve(links.size());
    double own_error = -1.0;
    for (std::size_t k = 0; k < links.size(); ++k) {
        const auto &link = links[k];
        if (!m_objective.held_out(link.term))
            matched.push_back(m_objective.match(candidate, state[link.other], link.term));
        else if (!held_out_known.empty())
            matched.push_back(held_out_known[k]);
        else
            matched.push_back(m_objective.held_out_match(i, link, candidate, own_error));
    }
    return matched;
}

JoinState::Proposal JoinState::propose(std::size_t i, std::vector<char> matched) const {
    Proposal proposal{i, 0.0, std::move(matched)};
    const auto links = m_objective.links(i);
    if (proposal.matched.size() != links.size())
        throw std::invalid_argument("join proposal: outcome count does not match links");
    // The hits/found changes the new outcomes cause on each side
    std::unordered_map<std::size_t, std::pair<long, long>> change; // t -> (hits, found)
    for (std::size_t k = 0; k < links.size(); ++k) {
        const auto &link = links[k];
        const bool now = m_matched[slot(i, link)];
        const bool then = proposal.matched[k];
        if (now == then)
            continue;
        const long delta = then ? 1 : -1;
        const std::array<std::size_t, 2> touched{count(link.other, link), count(i, link)};
        for (const auto t : std::span(touched).first(sides(link))) {
            auto &[hits, found] = change[t];
            found += delta;
            hits += link.positive ? delta : 0;
        }
    }
    if (change.empty())
        return proposal;
    const std::size_t n = m_objective.db().size();
    std::array<double, kMaxJoinTerms> sum{};
    std::array<long, kMaxJoinTerms> n_counted{};
    std::ranges::copy(m_sum, sum.begin());
    std::ranges::copy(m_counted, n_counted.begin());
    for (const auto &[t, counts] : change) {
        const auto old_found = m_found[t], old_hits = m_hits[t];
        const auto new_found =
            static_cast<std::size_t>(static_cast<long>(old_found) + counts.second);
        const auto new_hits = static_cast<std::size_t>(static_cast<long>(old_hits) + counts.first);
        if (counted(t, old_found)) {
            sum[t / n] -= f1(t, old_hits, old_found);
            --n_counted[t / n];
        }
        if (counted(t, new_found)) {
            sum[t / n] += f1(t, new_hits, new_found);
            ++n_counted[t / n];
        }
    }
    const auto terms = m_objective.join_terms();
    const double after =
        mean_f1(m_objective, std::span(sum).first(terms), std::span(n_counted).first(terms));
    proposal.gain = m_objective.scale(Objective::Join) * (after - metric());
    return proposal;
}

void JoinState::apply(const Proposal &proposal) {
    const auto i = proposal.trajectory;
    const auto links = m_objective.links(i);
    if (proposal.matched.size() != links.size())
        throw std::invalid_argument("join proposal: outcome count does not match links");
    for (std::size_t k = 0; k < links.size(); ++k) {
        const auto &link = links[k];
        const bool now = m_matched[slot(i, link)];
        const bool then = proposal.matched[k];
        if (now == then)
            continue;
        m_matched[slot(i, link)] = then;
        const long delta = then ? 1 : -1;
        const std::array<std::size_t, 2> touched{count(link.other, link), count(i, link)};
        for (const auto t : std::span(touched).first(sides(link))) {
            const auto term = t / m_objective.db().size();
            if (counted(t, m_found[t])) {
                m_sum[term] -= f1(t, m_hits[t], m_found[t]);
                --m_counted[term];
            }
            m_found[t] = static_cast<std::size_t>(static_cast<long>(m_found[t]) + delta);
            if (link.positive)
                m_hits[t] = static_cast<std::size_t>(static_cast<long>(m_hits[t]) + delta);
            if (counted(t, m_found[t])) {
                m_sum[term] += f1(t, m_hits[t], m_found[t]);
                ++m_counted[term];
            }
        }
    }
}

KnnState::KnnState(const Objective &objective, const KeptSets &state,
                   std::span<const KeptSets> watch_states, std::size_t watch)
    : m_objective(objective), m_watch(watch) {
    const auto &db = objective.db();
    const auto &workloads = objective.workloads();
    const std::size_t n = db.size();
    m_list.resize(n);
    m_distance.resize(n);
    m_hits.assign(n, 0);
    m_watchers.resize(n);
    std::vector<core::TrajectoryDb> compressed;
    compressed.push_back(core::materialize(db, state));
    for (const auto &watch_state : watch_states)
        compressed.push_back(core::materialize(db, watch_state));
    core::parallel_for(n, [&](std::size_t e) {
        const auto &entry = workloads.knn[e];
        if (!entry.valid)
            return;
        auto &list = m_list[e];
        list.assign(entry.top.begin(), entry.top.end());
        for (const auto &compressed_db : compressed)
            for (const auto j :
                 query::knn(compressed_db,
                            objective.config().compressed_knn_query ? compressed_db.trajectory(e)
                                                                    : db.trajectory(e),
                            std::min(watch + 1, n), workloads.frechet))
                if (j != e && std::ranges::find(list, j) == list.end())
                    list.push_back(static_cast<std::uint32_t>(j));
    });
    for (std::size_t e = 0; e < n; ++e) {
        m_valid += workloads.knn[e].valid;
        m_distance[e].assign(m_list[e].size(), 0.0);
        for (std::size_t p = 0; p < m_list[e].size(); ++p)
            m_watchers[m_list[e][p]].push_back(
                {static_cast<std::uint32_t>(e), static_cast<std::uint32_t>(p)});
    }
    const PreparedTrajectories prepared(db, state);
    const auto views = prepared.views();
    if (objective.config().compressed_knn_query) {
        core::parallel_for(n, [&](std::size_t e) { m_distance[e] = own_row(e, views[e], views); });
    } else {
        std::vector<std::vector<double>> own(n);
        core::parallel_for(n, [&](std::size_t i) { own[i] = distances(i, views[i]); });
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t watch_state = 0; watch_state < m_watchers[i].size(); ++watch_state)
                m_distance[m_watchers[i][watch_state].query][m_watchers[i][watch_state].position] =
                    own[i][watch_state];
    }
    m_rank.resize(n);
    core::parallel_for(n, [&](std::size_t e) {
        if (!m_list[e].empty())
            refresh(e);
    });
    m_hit_sum = std::accumulate(m_hits.begin(), m_hits.end(), 0ul);
}

std::vector<double> KnnState::distances(std::size_t i, const core::TrajectoryView &candidate,
                                        bool parallel) const {
    const auto &db = m_objective.db();
    const auto &watchers = m_watchers[i];
    std::vector<double> result(watchers.size());
    auto one = [&](std::size_t s) {
        result[s] = query::knn_distance(db.trajectory(watchers[s].query), candidate,
                                        m_objective.workloads().frechet);
    };
    if (parallel)
        core::parallel_for(watchers.size(), one);
    else
        for (std::size_t s = 0; s < watchers.size(); ++s)
            one(s);
    return result;
}

bool KnnState::active(std::size_t i) const {
    return watched(i) || (m_objective.config().compressed_knn_query && !m_list[i].empty());
}

std::vector<double> KnnState::own_row(std::size_t e, const core::TrajectoryView &query,
                                      std::span<const core::TrajectoryView> state) const {
    std::vector<double> row;
    row.reserve(m_list[e].size());
    for (const auto j : m_list[e])
        row.push_back(query::knn_distance(query, state[j], m_objective.workloads().frechet));
    return row;
}

KnnState::Move KnnState::move(std::size_t i, const core::TrajectoryView &candidate,
                              std::span<const core::TrajectoryView> state) const {
    if (!m_objective.config().compressed_knn_query)
        return {i, distances(i, candidate), {}};
    Move result{i, {}, own_row(i, candidate, state)};
    result.watched.reserve(m_watchers[i].size());
    for (const auto &watcher : m_watchers[i])
        result.watched.push_back(
            query::knn_distance(state[watcher.query], candidate, m_objective.workloads().frechet));
    return result;
}

double KnnState::propose(const Move &move) const {
    long change = 0;
    for (std::size_t s = 0; s < m_watchers[move.trajectory].size(); ++s) {
        const auto &watcher = m_watchers[move.trajectory][s];
        change += static_cast<long>(hits(watcher.query, watcher.position, move.watched[s])) -
                  static_cast<long>(m_hits[watcher.query]);
    }
    if (!move.own.empty() && !m_list[move.trajectory].empty())
        change += static_cast<long>(hits(move.trajectory, move.own)) -
                  static_cast<long>(m_hits[move.trajectory]);
    return m_objective.scale(Objective::Knn) * static_cast<double>(change) /
           static_cast<double>(m_objective.workloads().k * m_valid);
}

void KnnState::apply(const Move &move) {
    apply(move.trajectory, move.watched);
    if (move.own.empty() || m_list[move.trajectory].empty())
        return;
    const auto i = move.trajectory;
    m_distance[i] = move.own;
    m_hit_sum -= m_hits[i];
    refresh(i);
    m_hit_sum += m_hits[i];
}

std::size_t KnnState::hits(std::size_t e, std::span<const double> row) const {
    std::vector<std::pair<double, std::uint32_t>> order(m_list[e].size());
    for (std::size_t p = 0; p < order.size(); ++p)
        order[p] = {row[p], static_cast<std::uint32_t>(p)};
    return ranked_hits(e, order);
}

bool KnnState::ranks_before(std::size_t e, double distance, std::size_t position,
                            std::size_t other) const {
    return query::knn_rank_less(distance, m_list[e][position], m_distance[e][other],
                                m_list[e][other]);
}

// Only position's entry changes, so the top k changes by at most one swap with the k-th or the
// (k + 1)-th entry of the current ranking
std::size_t KnnState::hits(std::size_t e, std::size_t position, double distance) const {
    const std::size_t k = m_objective.workloads().k;
    const auto &rank = m_rank[e];
    if (m_list[e].size() <= k)
        return m_hits[e];
    const std::size_t members = m_objective.workloads().knn[e].top.size();
    const auto member = [&](std::size_t p) -> long { return p < members; };
    const auto current = static_cast<long>(m_hits[e]);
    const auto in_top = std::find(rank.begin(), rank.begin() + static_cast<std::ptrdiff_t>(k),
                                  static_cast<std::uint32_t>(position));
    if (in_top != rank.begin() + static_cast<std::ptrdiff_t>(k)) {
        const std::size_t next = rank[k];
        const std::size_t winner = ranks_before(e, distance, position, next) ? position : next;
        return static_cast<std::size_t>(current - member(position) + member(winner));
    }
    const std::size_t last = rank[k - 1];
    if (ranks_before(e, distance, position, last))
        return static_cast<std::size_t>(current - member(last) + member(position));
    return m_hits[e];
}

void KnnState::refresh(std::size_t e) {
    const std::size_t size = m_list[e].size();
    const std::size_t keep = std::min(m_objective.workloads().k + 1, size);
    std::vector<std::uint32_t> order(size);
    std::iota(order.begin(), order.end(), 0u);
    std::ranges::partial_sort(order, order.begin() + static_cast<std::ptrdiff_t>(keep),
                              [&](std::uint32_t a, std::uint32_t b) {
                                  return query::knn_rank_less(m_distance[e][a], m_list[e][a],
                                                              m_distance[e][b], m_list[e][b]);
                              });
    order.resize(keep);
    m_rank[e] = std::move(order);
    const std::size_t members = m_objective.workloads().knn[e].top.size();
    const std::size_t top = std::min(m_objective.workloads().k, size);
    m_hits[e] = static_cast<std::size_t>(
        std::count_if(m_rank[e].begin(), m_rank[e].begin() + static_cast<std::ptrdiff_t>(top),
                      [&](std::uint32_t p) { return p < members; }));
}

std::size_t KnnState::ranked_hits(std::size_t e,
                                  std::vector<std::pair<double, std::uint32_t>> &order) const {
    const auto &list = m_list[e];
    const std::size_t k = std::min(m_objective.workloads().k, list.size());
    const std::size_t members = m_objective.workloads().knn[e].top.size();
    // Ties go to the lower database index, like query::knn
    std::ranges::partial_sort(
        order, order.begin() + static_cast<std::ptrdiff_t>(k), [&](const auto &a, const auto &b) {
            return query::knn_rank_less(a.first, list[a.second], b.first, list[b.second]);
        });
    std::size_t count = 0;
    for (std::size_t p = 0; p < k; ++p)
        count += order[p].second < members;
    return count;
}

void KnnState::watch(std::span<const std::vector<core::TrajectoryView>> versions,
                     std::span<const core::TrajectoryView> state) {
    if (m_objective.config().compressed_knn_query)
        throw std::logic_error("knn state: watch needs original queries");
    const auto &db = m_objective.db();
    const auto frechet = m_objective.workloads().frechet;
    const std::size_t n = db.size();
    std::vector<double> radius(n, -1.0);
    core::parallel_for(n, [&](std::size_t e) {
        if (m_list[e].empty())
            return;
        auto distances = m_distance[e];
        const auto at = distances.begin() +
                        static_cast<std::ptrdiff_t>(std::min(m_watch, distances.size()) - 1);
        std::ranges::nth_element(distances, at);
        radius[e] = *at;
    });
    // A version keeps i's endpoints, so the endpoint bound rules out most queries for all of them
    std::vector<std::vector<std::pair<std::uint32_t, double>>> added(n);
    core::parallel_for(n, [&](std::size_t i) {
        if (versions[i].empty())
            return;
        const auto original = db.trajectory(i);
        for (std::size_t e = 0; e < n; ++e) {
            if (radius[e] < 0 || e == i ||
                query::endpoint_lower_bound(db.trajectory(e), original) > radius[e] ||
                std::ranges::any_of(m_watchers[i], [&](const Watcher &w) { return w.query == e; }))
                continue;
            for (const auto &version : versions[i])
                if (query::frechet_within(db.trajectory(e), version, radius[e], frechet)) {
                    added[i].push_back({static_cast<std::uint32_t>(e),
                                        query::knn_distance(db.trajectory(e), state[i], frechet)});
                    break;
                }
        }
    });
    std::vector<char> touched(n, 0);
    for (std::size_t i = 0; i < n; ++i)
        for (const auto &[e, distance] : added[i]) {
            m_watchers[i].push_back({e, static_cast<std::uint32_t>(m_list[e].size())});
            m_list[e].push_back(static_cast<std::uint32_t>(i));
            m_distance[e].push_back(distance);
            touched[e] = 1;
        }
    for (std::size_t e = 0; e < n; ++e)
        if (touched[e]) {
            m_hit_sum -= m_hits[e];
            refresh(e);
            m_hit_sum += m_hits[e];
        }
}

double KnnState::propose_bounded(std::size_t i, const core::TrajectoryView &candidate) const {
    if (m_objective.config().compressed_knn_query)
        throw std::logic_error("knn state: propose_bounded needs original queries");
    const auto &db = m_objective.db();
    const auto frechet = m_objective.workloads().frechet;
    const std::size_t k = m_objective.workloads().k;
    // Clear of the k-th place by more than the stored distances' error
    constexpr double margin = 2.5 * kKnnDistanceError;
    long change = 0;
    std::vector<double> others;
    for (const auto &[e, position] : m_watchers[i]) {
        const auto inside = hits(e, position, -1.0);
        const auto outside = hits(e, position, std::numeric_limits<double>::infinity());
        std::size_t now = inside;
        if (inside != outside) {
            // The k-th distance among the others: i is in clearly below it, out clearly above
            others.clear();
            for (std::size_t p = 0; p < m_list[e].size(); ++p)
                if (p != position)
                    others.push_back(m_distance[e][p]);
            const auto kth =
                others.begin() + static_cast<std::ptrdiff_t>(std::min(k, others.size()) - 1);
            std::ranges::nth_element(others, kth);
            const auto query = db.trajectory(e);
            if (*kth - margin > 0 &&
                query::frechet_within(query, candidate, *kth - margin, frechet))
                now = inside;
            else if (!query::frechet_within(query, candidate, *kth + margin, frechet))
                now = outside;
            else
                now = hits(e, position, query::knn_distance(query, candidate, frechet));
        }
        change += static_cast<long>(now) - static_cast<long>(m_hits[e]);
    }
    return m_objective.scale(Objective::Knn) * static_cast<double>(change) /
           static_cast<double>(k * m_valid);
}

double KnnState::propose(std::size_t i, std::span<const double> distances) const {
    long change = 0;
    for (std::size_t s = 0; s < m_watchers[i].size(); ++s) {
        const auto &watcher = m_watchers[i][s];
        change += static_cast<long>(hits(watcher.query, watcher.position, distances[s])) -
                  static_cast<long>(m_hits[watcher.query]);
    }
    return m_objective.scale(Objective::Knn) * static_cast<double>(change) /
           static_cast<double>(m_objective.workloads().k * m_valid);
}

void KnnState::apply(std::size_t i, std::span<const double> distances) {
    for (std::size_t s = 0; s < m_watchers[i].size(); ++s) {
        const auto &watcher = m_watchers[i][s];
        m_distance[watcher.query][watcher.position] = distances[s];
        m_hit_sum -= m_hits[watcher.query];
        refresh(watcher.query);
        m_hit_sum += m_hits[watcher.query];
    }
}

double KnnState::metric() const {
    return static_cast<double>(m_hit_sum) /
           static_cast<double>(m_objective.workloads().k * m_valid);
}

} // namespace teacher
