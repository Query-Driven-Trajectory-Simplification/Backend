#include "backend/engine/teacher/objective.hpp"

#include "backend/engine/core/interpolation.hpp"
#include "backend/engine/core/parallel.hpp"
#include "backend/engine/core/simplification.hpp"
#include "backend/engine/query/frechet_helper.hpp"
#include "backend/engine/query/knn.hpp"

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

bool inside(const query::RangeBox &box, const core::TrajectoryView &trajectory, std::size_t p) {
    return trajectory.t[p] >= box.t_min && trajectory.t[p] <= box.t_max &&
           trajectory.x[p] >= box.x_min && trajectory.x[p] <= box.x_max &&
           trajectory.y[p] >= box.y_min && trajectory.y[p] <= box.y_max;
}

constexpr std::size_t kMaxJoinTerms = 8;

double f1_of(std::size_t hits, std::size_t found, std::size_t expected) {
    return 2.0 * static_cast<double>(hits) / static_cast<double>(found + expected);
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
    // Extra thresholds use every pair, which would train on the pairs held back as unseen
    if (!config.all_join_pairs && !workloads.extra_joins.empty())
        throw std::invalid_argument(
            "objective: seen-pairs-only training cannot use extra join thresholds");

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
            m_range[i].push_back(
                {static_cast<std::uint32_t>(b),
                 1.0 / (static_cast<double>(ref.size()) * static_cast<double>(scored))});
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
        if (!config.all_join_pairs && !join_pair.seen)
            return;
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
    m_with_truth = static_cast<std::size_t>(
        std::ranges::count_if(m_truth, [](auto count) { return count > 0; }));

    // Floors: endpoints only, partners included, like the J4 floor
    const auto endpoints = core::endpoints_only(db);
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
        for (const auto m : {Knn, Range, WhereAt})
            m_endpoint_separable[i] += m_scale[m] * endpoint_terms[m];
    });
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

    for (const auto &[b, weight] : m_range[i]) {
        const auto &box = m_workloads.range.queries[b].box;
        for (std::size_t k = 0; k < candidate.size(); ++k)
            if (inside(box, candidate, k)) {
                terms[Range] += weight;
                break;
            }
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
    return m_scale[Knn] * t[Knn] + m_scale[Range] * t[Range] + m_scale[WhereAt] * t[WhereAt] -
           m_endpoint_separable[i];
}

double Objective::independent_value(std::size_t i, const core::TrajectoryView &candidate,
                                    std::span<const core::TrajectoryView> partners,
                                    double beat) const {
    return independent_value(prepare_independent(i, candidate, partners), beat);
}

Objective::IndependentScore
Objective::prepare_independent(std::size_t i, const core::TrajectoryView &candidate,
                               std::span<const core::TrajectoryView> partners) const {
    IndependentScore score;
    score.trajectory = i;
    score.candidate = candidate;
    score.terms = raw_separable(i, candidate, false);
    std::array<std::size_t, kMaxJoinTerms> hits{}, found{};
    for (const auto &link : m_links[i]) {
        // Held out: the partner is the original query and i the database side
        const bool matched =
            m_term_held_out[link.term]
                ? match(m_db.trajectory(link.other), candidate, link.term)
                : match(candidate,
                        partners.empty() ? m_db.trajectory(link.other) : partners[link.other],
                        link.term);
        found[link.term] += matched;
        hits[link.term] += matched && link.positive;
    }
    double f1_sum = 0;
    for (std::size_t k = 0; k < m_eps.size(); ++k)
        if (found[k] + m_truth[slot(i, k)] != 0)
            f1_sum += f1_of(hits[k], found[k], m_truth[slot(i, k)]);
    score.join =
        m_with_truth != 0 ? m_scale[Join] * f1_sum / static_cast<double>(m_with_truth) : 0.0;
    score.bound = independent_sum(score, score.terms[Knn]) +
                  m_scale[Knn] * m_knn_weight * static_cast<double>(m_knn[i].size());
    return score;
}

double Objective::independent_sum(const IndependentScore &score, double knn) const {
    // Same sums in the same order as separable, so the value is identical to the last bit
    return m_scale[Knn] * knn + m_scale[Range] * score.terms[Range] +
           m_scale[WhereAt] * score.terms[WhereAt] - m_endpoint_separable[score.trajectory] +
           score.join;
}

double Objective::independent_value(const IndependentScore &score, double beat) const {
    const auto i = score.trajectory;
    double knn = score.terms[Knn];
    auto value = [&] { return independent_sum(score, knn); };
    // Stop once even all remaining hits cannot beat beat
    const double per_hit = m_scale[Knn] * m_knn_weight;
    std::size_t left = m_knn[i].size();
    for (const auto &m : m_knn[i]) {
        if (value() + per_hit * static_cast<double>(left) < beat - kBoundSlack)
            return value();
        --left;
        if (query::frechet_within(m_db.trajectory(m.query), score.candidate, m.cutoff,
                                  m_workloads.frechet))
            knn += m_knn_weight;
    }
    return value();
}

double Objective::independent_bound(std::size_t i, const core::TrajectoryView &candidate,
                                    std::span<const core::TrajectoryView> partners) const {
    return prepare_independent(i, candidate, partners).bound;
}

double Objective::join_metric(std::span<const core::TrajectoryView> state) const {
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
    double sum = 0;
    std::size_t counted = 0;
    for (std::size_t t = 0; t < m_truth.size(); ++t)
        if (found[t] + m_truth[t] != 0) {
            sum += f1_of(hits[t], found[t], m_truth[t]);
            ++counted;
        }
    return counted ? sum / static_cast<double>(counted) : 0.0;
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
        for (const auto m : {Knn, Range, WhereAt})
            total[m] += terms[m];
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
        auto found = query::knn(compressed, m_db.trajectory(e), std::min(k + 1, m_db.size()),
                                m_workloads.frechet);
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

JoinState::JoinState(const Objective &objective, std::span<const core::TrajectoryView> state)
    : m_objective(objective) {
    const std::size_t n = state.size();
    m_matched.assign(2 * objective.pair_count(), 0);
    m_hits.assign(objective.slots(), 0);
    m_found.assign(objective.slots(), 0);
    // Each pair once, from its smaller side. Held out: each direction from its compressed side
    std::vector<std::vector<std::pair<std::size_t, char>>> outcome(n);
    core::parallel_for(n, [&](std::size_t i) {
        for (const auto &link : objective.links(i)) {
            if (objective.held_out(link.term))
                outcome[i].push_back(
                    {slot(i, link),
                     objective.match(objective.db().trajectory(link.other), state[i], link.term)});
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

JoinState::Proposal JoinState::propose(std::size_t i, const core::TrajectoryView &candidate,
                                       std::span<const core::TrajectoryView> state) const {
    Proposal proposal{i, 0.0, {}};
    proposal.matched.reserve(m_objective.links(i).size());
    // New outcomes of i's pairs, and the hits/found changes they cause on each side
    std::unordered_map<std::size_t, std::pair<long, long>> change; // t -> (hits, found)
    for (const auto &link : m_objective.links(i)) {
        const bool now = m_matched[slot(i, link)];
        const bool then =
            m_objective.held_out(link.term)
                ? m_objective.match(m_objective.db().trajectory(link.other), candidate, link.term)
                : m_objective.match(candidate, state[link.other], link.term);
        proposal.matched.push_back(then);
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
    double sum = m_sum;
    long n_counted = static_cast<long>(m_counted);
    for (const auto &[t, counts] : change) {
        const auto old_found = m_found[t], old_hits = m_hits[t];
        const auto new_found =
            static_cast<std::size_t>(static_cast<long>(old_found) + counts.second);
        const auto new_hits = static_cast<std::size_t>(static_cast<long>(old_hits) + counts.first);
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
            if (counted(t, m_found[t])) {
                m_sum -= f1(t, m_hits[t], m_found[t]);
                --m_counted;
            }
            m_found[t] = static_cast<std::size_t>(static_cast<long>(m_found[t]) + delta);
            if (link.positive)
                m_hits[t] = static_cast<std::size_t>(static_cast<long>(m_hits[t]) + delta);
            if (counted(t, m_found[t])) {
                m_sum += f1(t, m_hits[t], m_found[t]);
                ++m_counted;
            }
        }
    }
}

KnnState::KnnState(const Objective &objective, const KeptSets &state,
                   std::span<const KeptSets> watch_states, std::size_t watch)
    : m_objective(objective) {
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
            for (const auto j : query::knn(compressed_db, db.trajectory(e), std::min(watch + 1, n),
                                           workloads.frechet))
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
    std::vector<std::vector<double>> own(n);
    core::parallel_for(n, [&](std::size_t i) { own[i] = distances(i, views[i]); });
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t watch_state = 0; watch_state < m_watchers[i].size(); ++watch_state)
            m_distance[m_watchers[i][watch_state].query][m_watchers[i][watch_state].position] =
                own[i][watch_state];
    core::parallel_for(n, [&](std::size_t e) {
        if (!m_list[e].empty())
            m_hits[e] = hits(e, m_list[e].size(), 0.0);
    });
    m_hit_sum = std::accumulate(m_hits.begin(), m_hits.end(), 0ul);
}

std::vector<double> KnnState::distances(std::size_t i,
                                        const core::TrajectoryView &candidate) const {
    const auto &db = m_objective.db();
    std::vector<double> result;
    result.reserve(m_watchers[i].size());
    for (const auto &watcher : m_watchers[i])
        result.push_back(query::frechet_distance(db.trajectory(watcher.query), candidate,
                                                 m_objective.workloads().frechet));
    return result;
}

std::size_t KnnState::hits(std::size_t e, std::size_t position, double distance) const {
    const auto &list = m_list[e];
    const std::size_t k = std::min(m_objective.workloads().k, list.size());
    const std::size_t members = m_objective.workloads().knn[e].top.size();
    std::vector<std::pair<double, std::uint32_t>> order(list.size());
    for (std::size_t p = 0; p < list.size(); ++p)
        order[p] = {p == position ? distance : m_distance[e][p], static_cast<std::uint32_t>(p)};
    // Ties go to the lower database index, like query::knn
    std::ranges::partial_sort(
        order, order.begin() + static_cast<std::ptrdiff_t>(k), [&](const auto &a, const auto &b) {
            return a.first < b.first || (a.first == b.first && list[a.second] < list[b.second]);
        });
    std::size_t count = 0;
    for (std::size_t p = 0; p < k; ++p)
        count += order[p].second < members;
    return count;
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
        m_hits[watcher.query] = hits(watcher.query, m_list[watcher.query].size(), 0.0);
        m_hit_sum += m_hits[watcher.query];
    }
}

double KnnState::metric() const {
    return static_cast<double>(m_hit_sum) /
           static_cast<double>(m_objective.workloads().k * m_valid);
}

} // namespace teacher
