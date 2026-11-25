#include "backend/engine/teacher/objective.hpp"

#include "backend/engine/core/parallel.hpp"
#include "backend/engine/core/simplification.hpp"
#include "backend/engine/query/frechet_helper.hpp"
#include "backend/engine/query/knn.hpp"
#include "backend/engine/query/range_query.hpp"
#include "backend/engine/selection/rdp.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>

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
// Deviation of the coarse copies that pre-screen tests against originals
constexpr double kCoarseM = 5.0;

// Largest distance from a vertex of a to the polyline b, squared: a lower bound on the continuous
// Fréchet distance. Each vertex searches outward from the previous vertex's nearest segment and
// stops once one is no farther than the running maximum
double squared_directed_hausdorff(const core::TrajectoryView &a, const core::TrajectoryView &b) {
    const auto segment = [&](std::size_t i, std::size_t j) {
        const double ax = b.x[j], ay = b.y[j], dx = b.x[j + 1] - ax, dy = b.y[j + 1] - ay;
        const double px = a.x[i] - ax, py = a.y[i] - ay;
        const double length = dx * dx + dy * dy;
        const double along = px * dx + py * dy;
        if (length == 0.0 || along <= 0.0)
            return px * px + py * py;
        if (along >= length) {
            const double ex = a.x[i] - b.x[j + 1], ey = a.y[i] - b.y[j + 1];
            return ex * ex + ey * ey;
        }
        const double cross = px * dy - py * dx;
        return cross * cross / length;
    };
    if (a.size() == 0 || b.size() == 0)
        return std::numeric_limits<double>::infinity();
    double bound = 0.0;
    if (b.size() == 1) {
        for (std::size_t i = 0; i < a.size(); ++i) {
            const double dx = a.x[i] - b.x[0], dy = a.y[i] - b.y[0];
            bound = std::max(bound, dx * dx + dy * dy);
        }
        return bound;
    }
    const std::size_t segments = b.size() - 1;
    std::size_t start = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double nearest = segment(i, start);
        std::size_t best = start;
        for (std::size_t step = 1; nearest > bound && (step <= start || start + step < segments);
             ++step) {
            if (step <= start) {
                const double d = segment(i, start - step);
                if (d < nearest) {
                    nearest = d;
                    best = start - step;
                }
            }
            if (start + step < segments) {
                const double d = segment(i, start + step);
                if (d < nearest) {
                    nearest = d;
                    best = start + step;
                }
            }
        }
        bound = std::max(bound, nearest);
        start = best;
    }
    return bound;
}

} // namespace

// Upper bound on the continuous Fréchet distance between original and a simplification keeping
// a subsequence of its points: each kept segment walks its original stretch at the same share
// of arc length. Along each original edge the gap is convex, so its maximum is at a vertex
double simplification_bound(const core::TrajectoryView &original, const core::TrajectoryView &kept,
                            std::span<const double> segments) {
    const auto edge = [&](std::size_t j) {
        return segments.empty() ? std::hypot(original.x[j + 1] - original.x[j],
                                             original.y[j + 1] - original.y[j])
                                : segments[j];
    };
    const std::size_t n = original.size();
    if (kept.size() < 2 || n < 2)
        return std::numeric_limits<double>::infinity();
    double bound_sq = 0.0;
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
            length += edge(j);
        double walked = 0.0;
        const double kx = kept.x[k], ky = kept.y[k];
        const double kdx = kept.x[k + 1] - kx, kdy = kept.y[k + 1] - ky;
        for (std::size_t j = at; j <= end; ++j) {
            if (j > at)
                walked += edge(j - 1);
            const double u = length > 0 ? walked / length : 0.0;
            const double dx = original.x[j] - (kx + u * kdx);
            const double dy = original.y[j] - (ky + u * kdy);
            bound_sq = std::max(bound_sq, dx * dx + dy * dy);
        }
        at = end;
    }
    return std::sqrt(bound_sq);
}

namespace {

// Cached kNN distances are upper bounds on a fixed millimetre grid.
constexpr double kKnnDistanceError = query::kKnnResolutionM;
// Ranked neighbours store knn_distance (upper bound on the 1 mm grid); twice that brackets the
// true continuous distance from below
constexpr double kRankedDistanceError = 2e-3;

// The shortest RDP prefix of original within max_error, if it at least halves the points
std::optional<std::vector<std::size_t>> coarse_copy(const core::TrajectoryView &original,
                                                    double max_error) {
    if (original.size() < 8)
        return std::nullopt;
    const auto order = selection::rdp::sequence(original, original.size());
    auto prefix = [&](std::size_t b) {
        std::vector<std::size_t> kept(order.begin(),
                                      order.begin() + static_cast<std::ptrdiff_t>(b));
        std::ranges::sort(kept);
        return kept;
    };
    auto within = [&](std::size_t b) {
        return simplification_bound(original, PreparedTrajectory(original, prefix(b)).view()) <=
               max_error;
    };
    // Doubling, then bisection between the last miss and the first hit
    const std::size_t most = original.size() / 2;
    std::size_t lo = 2, hi = 2;
    while (!within(hi)) {
        if (hi == most)
            return std::nullopt;
        lo = hi;
        hi = std::min(most, 2 * hi);
    }
    while (lo + 1 < hi) {
        const auto mid = (lo + hi) / 2;
        (within(mid) ? hi : lo) = mid;
    }
    return prefix(hi);
}

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
    if (!(config.min_shape_m < config.max_shape_m))
        throw std::invalid_argument("objective: min_shape_m must be below max_shape_m");
    m_shape_weight = n > 0 ? 1.0 / static_cast<double>(n) : 0.0;
    const double weight_sum = check_term_weights(config.term_weights);

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
    // A bracket on each membership's distance, from the ranked neighbours
    if (workloads.ranked.size() == n)
        for (std::size_t i = 0; i < n; ++i)
            for (auto &membership : m_knn[i])
                for (const auto &neighbour : workloads.ranked[membership.query])
                    if (neighbour.index == i) {
                        membership.upper = neighbour.distance;
                        membership.lower = std::max(0.0, neighbour.distance - kRankedDistanceError);
                        break;
                    }
    m_margins.resize(n);
    const std::size_t k = workloads.k;
    if (config.term_weights[Margin] > 0) {
        // Both sides of each query's k-th place: a true neighbour against the (k + 1)th, an
        // outsider against the k-th
        if (workloads.ranked.size() != n)
            throw std::invalid_argument("objective: the margin term needs ranked neighbours");
        for (std::size_t e = 0; e < n; ++e) {
            const auto &list = workloads.ranked[e];
            if (!workloads.knn[e].valid || list.size() <= k)
                continue;
            const double inside = list[k - 1].distance, outside = list[k].distance;
            for (std::size_t r = 0; r < list.size(); ++r)
                m_margins[list[r].index].push_back(
                    std::max(0.0, r < k ? outside - list[r].distance : list[r].distance - inside));
        }
    }
    std::size_t memberships = 0;
    for (const auto &margins : m_margins)
        memberships += margins.size();
    m_margin_weight = memberships > 0 ? 1.0 / static_cast<double>(memberships) : 0.0;

    // Range: invert each box's reference
    m_range.resize(n);
    for (const auto &query : workloads.range.queries)
        m_scored_boxes += !query.reference.empty();
    for (std::size_t b = 0; b < workloads.range.queries.size(); ++b) {
        const auto &ref = workloads.range.queries[b].reference;
        for (const auto i : ref)
            m_range[i].push_back({static_cast<std::uint32_t>(b),
                                  range_marginal(ref.size() - 1, ref.size(), m_scored_boxes)});
    }

    // Where-at: probes per trajectory, plus a time-sorted copy for linear scoring
    m_probes.resize(n);
    for (std::size_t k = 0; k < workloads.where_at.queries.size(); ++k)
        m_probes[workloads.where_at.queries[k].trajectory_index].push_back(
            static_cast<std::uint32_t>(k));
    m_sorted_probes.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        auto &sorted = m_sorted_probes[i];
        sorted.reserve(m_probes[i].size());
        for (const auto k : m_probes[i]) {
            const auto &probe = workloads.where_at.queries[k];
            sorted.push_back({probe.t, probe.reference.x, probe.reference.y});
        }
        std::ranges::sort(sorted, {}, &Probe::t);
    }
    const auto with_probes =
        std::ranges::count_if(m_probes, [](auto &probes) { return !probes.empty(); });
    m_where_at_weight = with_probes ? 1.0 / static_cast<double>(with_probes) : 0.0;

    // Join terms: join_eps_m, then the extra thresholds. Pairs are stored positives first, then
    // negatives
    struct TermPairs {
        const std::vector<JoinPair> *positives, *negatives;
    };
    std::vector<TermPairs> terms{{&workloads.positives, &workloads.negatives}};
    m_eps = {workloads.join_eps_m};
    for (const auto &extra : workloads.extra_joins) {
        terms.push_back({&extra.positives, &extra.negatives});
        m_eps.push_back(extra.eps_m);
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

    m_segments.resize(n);
    core::parallel_for(n, [&](std::size_t i) {
        const auto t = db.trajectory(i);
        auto &edges = m_segments[i];
        edges.resize(t.size() > 0 ? t.size() - 1 : 0);
        for (std::size_t j = 0; j < edges.size(); ++j)
            edges[j] = std::hypot(t.x[j + 1] - t.x[j], t.y[j + 1] - t.y[j]);
    });

    // Coarse copies, to pre-screen tests against originals
    m_coarse.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        m_coarse.emplace_back(db.trajectory(i), std::vector<std::size_t>{});
    m_coarse_error.assign(n, -1.0);
    if (m_workloads.frechet == query::FrechetVariant::Continuous)
        core::parallel_for(n, [&](std::size_t i) {
            const auto original = db.trajectory(i);
            if (const auto kept = coarse_copy(original, kCoarseM)) {
                m_coarse[i].assign(original, *kept);
                m_coarse_error[i] = simplification_bound(original, m_coarse[i].view());
            }
        });

    m_certify = m_workloads.frechet == query::FrechetVariant::Continuous;
    bracket_pairs();

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
        const auto endpoint_terms =
            raw_separable(i, PreparedTrajectory(db.trajectory(i), endpoints[i]).view());
        for (const auto m : {WhereAt, Shape, Margin})
            m_endpoint_separable[i] += m_scale[m] * endpoint_terms[m];
    });
}

void Objective::bracket_pairs() {
    // Each pair's originals, once per pair across terms: the discrete distance bounds the
    // continuous one from above (exact below the largest threshold), Hausdorff from below
    constexpr double inf = std::numeric_limits<double>::infinity();
    m_pair_lower.assign(m_pair_i.size(), 0.0);
    m_pair_upper.assign(m_pair_i.size(), inf);
    m_pair_distance.clear();
    if (m_config.pair_distances)
        m_pair_distance.assign(m_pair_i.size(), inf);
    if (!m_certify && !m_config.pair_distances)
        return;
    std::vector<std::uint32_t> order(m_pair_i.size());
    std::iota(order.begin(), order.end(), 0u);
    std::ranges::sort(order, [&](std::uint32_t a, std::uint32_t b) {
        return std::pair{m_pair_i[a], m_pair_j[a]} < std::pair{m_pair_i[b], m_pair_j[b]};
    });
    std::vector<std::size_t> starts;
    for (std::size_t r = 0; r < order.size(); ++r)
        if (r == 0 || m_pair_i[order[r]] != m_pair_i[order[r - 1]] ||
            m_pair_j[order[r]] != m_pair_j[order[r - 1]])
            starts.push_back(r);
    const double largest = *std::ranges::max_element(m_eps);
    const double cutoff = largest * largest;
    core::parallel_for(starts.size(), [&](std::size_t s) {
        const auto first = order[starts[s]];
        const auto a = m_db.trajectory(m_pair_i[first]), b = m_db.trajectory(m_pair_j[first]);
        std::vector<double> prev, curr;
        const std::size_t end = s + 1 < starts.size() ? starts[s + 1] : order.size();
        if (!m_certify) {
            // Discrete: the distance itself, with no cutoff
            const double distance =
                std::sqrt(query::get_squared_frechet_distance(a, b, inf, prev, curr));
            for (std::size_t r = starts[s]; r < end; ++r)
                m_pair_distance[order[r]] = distance;
            return;
        }
        const double squared = query::get_squared_frechet_distance(a, b, cutoff, prev, curr);
        const double upper = squared < cutoff ? std::sqrt(squared) : inf;
        const double lower = std::sqrt(std::max(
            {query::squared_endpoint_lower_bound(a, b), query::squared_box_extent_lower_bound(a, b),
             squared_directed_hausdorff(a, b), squared_directed_hausdorff(b, a)}));
        for (std::size_t r = starts[s]; r < end; ++r) {
            m_pair_lower[order[r]] = lower;
            m_pair_upper[order[r]] = upper;
        }
        if (!m_config.pair_distances)
            return;
        // The continuous distance searched within the bracket. Above the largest threshold the
        // discrete distance, now without a cutoff, is still an upper bound
        const double search_upper =
            std::isfinite(upper)
                ? upper
                : std::sqrt(query::get_squared_frechet_distance(a, b, inf, prev, curr));
        const double distance = query::continuous_frechet_distance(a, b, lower, search_upper);
        for (std::size_t r = starts[s]; r < end; ++r)
            m_pair_distance[order[r]] = distance;
    });
}

double Objective::pair_distance(std::size_t pair) const {
    if (m_pair_distance.empty())
        throw std::logic_error("objective: pair distances need ObjectiveConfig::pair_distances");
    return m_pair_distance[pair];
}

double Objective::slack(const JoinLink &link) const {
    return std::abs(pair_distance(link.pair) - m_eps[link.term]);
}

double flip_probability(double slack, double deviation, double bandwidth) {
    if (deviation <= 0.0)
        return 0.0;
    if (std::isinf(deviation))
        return 0.5;
    // With slack x the outcome flips when the distance moves past x the wrong way:
    // max(0, deviation - x) / (2 deviation). Averaged over x uniform in [a, b]
    const double a = std::max(0.0, slack - bandwidth), b = slack + bandwidth;
    if (b <= a)
        return std::max(0.0, deviation - slack) / (2.0 * deviation);
    // Antiderivative of max(0, deviation - x), flat from x = deviation on
    const auto integral = [deviation](double x) {
        x = std::min(x, deviation);
        return deviation * x - 0.5 * x * x;
    };
    return (integral(b) - integral(a)) / (2.0 * deviation * (b - a));
}

std::optional<bool> Objective::certify(double lower, double upper, double deviation,
                                       double cutoff) const {
    // |d(query, candidate) - d(query, original)| <= d(original, candidate) <= deviation
    if (upper + deviation <= cutoff * (1.0 - kCertificateMargin))
        return true;
    if (lower - deviation > cutoff * (1.0 + kCertificateMargin))
        return false;
    return std::nullopt;
}

bool Objective::held_out_match(std::size_t i, const JoinLink &link,
                               const core::TrajectoryView &candidate, double &own_error) const {
    if (m_certify) {
        if (own_error < 0)
            own_error = deviation(i, candidate);
        if (const auto known = certify(m_pair_lower[link.pair], m_pair_upper[link.pair], own_error,
                                       m_eps[link.term]))
            return *known;
    }
    return original_within(link.other, candidate, m_eps[link.term]);
}

bool Objective::original_within(std::size_t j, const core::TrajectoryView &candidate,
                                double radius) const {
    if (m_coarse_error[j] >= 0) {
        // d(original, coarse) <= error, so the coarse copy decides outside a band of 2 error
        const auto coarse = m_coarse[j].view();
        const double error = m_coarse_error[j];
        if (radius - error > 0 &&
            query::continuous_frechet_within(coarse, candidate,
                                             (radius - error) * (1.0 - kCertificateMargin)))
            return true;
        if (!query::continuous_frechet_within(coarse, candidate,
                                              (radius + error) * (1.0 + kCertificateMargin)))
            return false;
    }
    return query::frechet_within(m_db.trajectory(j), candidate, radius, m_workloads.frechet);
}

bool Objective::match(const core::TrajectoryView &a, const core::TrajectoryView &b,
                      std::size_t term) const {
    return query::frechet_within(a, b, m_eps[term], m_workloads.frechet);
}

Objective::Terms Objective::raw_separable(std::size_t i, const core::TrajectoryView &candidate,
                                          double deviation) const {
    Terms terms{};
    if (!m_sorted_probes[i].empty()) {
        const double far = m_config.max_tolerance_m * m_config.max_tolerance_m * (1.0 + 1e-12);
        const double near = m_config.min_tolerance_m * m_config.min_tolerance_m * (1.0 - 1e-12);
        const std::size_t last = candidate.size() - 1;
        const double t_first = static_cast<double>(candidate.t[0]);
        const double t_last = static_cast<double>(candidate.t[last]);
        std::size_t after = 1;
        double sum = 0.0;
        for (const auto &probe : m_sorted_probes[i]) {
            double x, y;
            if (probe.t <= t_first || last == 0) {
                x = candidate.x[0];
                y = candidate.y[0];
            } else if (probe.t >= t_last) {
                x = candidate.x[last];
                y = candidate.y[last];
            } else {
                while (after < last && static_cast<double>(candidate.t[after]) <= probe.t)
                    ++after;
                const std::size_t before = after - 1;
                const double t_before = static_cast<double>(candidate.t[before]);
                const double t_after = static_cast<double>(candidate.t[after]);
                const double fraction = (probe.t - t_before) / (t_after - t_before);
                x = candidate.x[before] + fraction * (candidate.x[after] - candidate.x[before]);
                y = candidate.y[before] + fraction * (candidate.y[after] - candidate.y[before]);
            }
            const double dx = x - probe.x, dy = y - probe.y;
            const double sq = dx * dx + dy * dy;
            if (sq <= near)
                sum += 1.0;
            else if (sq < far)
                sum += ramp(std::sqrt(sq), m_config.min_tolerance_m, m_config.max_tolerance_m);
        }
        terms[WhereAt] = m_where_at_weight * sum / static_cast<double>(m_sorted_probes[i].size());
    }

    const bool shape = m_config.term_weights[Shape] > 0;
    const bool margin = m_config.term_weights[Margin] > 0 && !m_margins[i].empty();
    if (shape || margin) {
        if (deviation < 0)
            deviation = this->deviation(i, candidate);
        if (shape)
            terms[Shape] =
                m_shape_weight * ramp(deviation, m_config.min_shape_m, m_config.max_shape_m);
        // The distance moves uniformly within +-deviation; a hit survives up to its margin
        if (margin)
            for (const double gap : m_margins[i])
                terms[Margin] +=
                    m_margin_weight * (deviation <= gap ? 1.0 : 0.5 * (1.0 + gap / deviation));
    }
    return terms;
}

double Objective::separable(std::size_t i, std::span<const std::size_t> kept) const {
    return separable(i, PreparedTrajectory(m_db.trajectory(i), kept).view());
}

double Objective::separable(std::size_t i, const core::TrajectoryView &candidate,
                            double deviation) const {
    const auto t = raw_separable(i, candidate, deviation);
    return m_scale[WhereAt] * t[WhereAt] + m_scale[Shape] * t[Shape] + m_scale[Margin] * t[Margin] -
           m_endpoint_separable[i];
}

std::vector<double> Objective::range_weights(std::size_t i,
                                             std::span<const core::TrajectoryView> partners) const {
    if (m_range[i].empty())
        return {};
    std::vector<double> weights;
    weights.reserve(m_range[i].size());
    for (const auto &membership : m_range[i]) {
        const auto &q = m_workloads.range.queries[membership.box];
        std::size_t others = 0;
        for (const auto j : q.reference)
            if (j != i && (partners.empty() || query::intersects(partners[j], q.box)))
                ++others;
        weights.push_back(range_marginal(others, q.reference.size(), m_scored_boxes));
    }
    return weights;
}

double Objective::independent_value(std::size_t i, const core::TrajectoryView &candidate,
                                    std::span<const core::TrajectoryView> partners, double beat,
                                    std::span<const double> range_weights,
                                    const JoinState *join_state) const {
    return independent_value(prepare_independent(i, candidate, partners, range_weights, join_state),
                             beat);
}

Objective::IndependentScore
Objective::prepare_independent(std::size_t i, const core::TrajectoryView &candidate,
                               std::span<const core::TrajectoryView> partners,
                               std::span<const double> range_weights,
                               const JoinState *join_state) const {
    IndependentScore score;
    score.trajectory = i;
    score.candidate = candidate;
    score.deviation = deviation(i, candidate);
    score.terms = raw_separable(i, candidate, score.deviation);
    if (!range_weights.empty() && range_weights.size() != m_range[i].size())
        throw std::invalid_argument("objective: one range weight per membership needed");
    if (m_scale[Range] > 0 && !m_range[i].empty()) {
        const auto owned_weights =
            range_weights.empty() ? this->range_weights(i, partners) : std::vector<double>{};
        if (range_weights.empty())
            range_weights = owned_weights;
        for (std::size_t k = 0; k < m_range[i].size(); ++k)
            if (query::intersects(candidate, m_workloads.range.queries[m_range[i][k].box].box))
                score.terms[Range] += range_weights[k];
    }
    if (m_scale[Join] > 0 && join_state) {
        double own_error = score.deviation;
        score.join = join_state->gain(i, candidate, own_error);
    }
    score.bound = independent_sum(score, 0.0) +
                  m_scale[Knn] * m_knn_weight * static_cast<double>(m_knn[i].size());
    return score;
}

double Objective::independent_sum(const IndependentScore &score, double knn) const {
    // Range is i's marginal F1 against fixed partners; the loop tracks the actual joint F1
    return m_scale[Knn] * knn + m_scale[Range] * score.terms[Range] +
           m_scale[WhereAt] * score.terms[WhereAt] + m_scale[Shape] * score.terms[Shape] +
           m_scale[Margin] * score.terms[Margin] - m_endpoint_separable[score.trajectory] +
           score.join;
}

double Objective::independent_value(const IndependentScore &score, double beat) const {
    if (m_scale[Knn] <= 0)
        return independent_sum(score, 0.0);
    double knn = 0.0;
    auto value = [&] { return independent_sum(score, knn); };
    // Stop once even all remaining hits cannot beat beat
    const double per_hit = m_scale[Knn] * m_knn_weight;
    std::size_t left = m_knn[score.trajectory].size();
    for (const auto &m : m_knn[score.trajectory]) {
        if (value() + per_hit * static_cast<double>(left) < beat - kBoundSlack)
            break;
        --left;
        const auto known =
            m_certify ? certify(m.lower, m.upper, score.deviation, m.cutoff) : std::nullopt;
        if (known ? *known : original_within(m.query, score.candidate, m.cutoff))
            knn += m_knn_weight;
    }
    return value();
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
    // Each direction from its compressed side, using pair brackets and coarse copies
    std::vector<std::vector<char>> outcome(state.size());
    core::parallel_for(state.size(), [&](std::size_t i) {
        const auto &links = m_links[i];
        outcome[i].reserve(links.size());
        double own_error = -1.0;
        for (const auto &link : links)
            outcome[i].push_back(held_out_match(i, link, state[i], own_error));
    });
    std::vector<std::size_t> hits(m_truth.size(), 0), found(m_truth.size(), 0);
    for (std::size_t i = 0; i < state.size(); ++i) {
        const auto &links = m_links[i];
        for (std::size_t k = 0; k < links.size(); ++k)
            if (outcome[i][k]) {
                const auto t = slot(links[k].other, links[k].term);
                ++found[t];
                hits[t] += links[k].positive;
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
    core::parallel_for(kept.size(), [&](std::size_t i) { per[i] = raw_separable(i, views[i]); });
    Terms total{};
    for (const auto &terms : per)
        for (const auto m : {WhereAt, Shape, Margin})
            total[m] += terms[m];
    total[Range] = RangeState(*this, views).metric();
    total[Join] = join_metric(views);
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

double Objective::deviation(std::size_t i, const core::TrajectoryView &candidate) const {
    return simplification_bound(m_db.trajectory(i), candidate, m_segments[i]);
}

double Objective::total(const KeptSets &kept) const { return total(metrics(kept)); }

double Objective::total(const Terms &values) const {
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

double RangeState::gain(std::size_t i, const core::TrajectoryView &candidate) const {
    if (!m_scored)
        return 0.0;
    const auto memberships = m_objective.boxes(i);
    double change = 0.0;
    for (std::size_t k = 0; k < memberships.size(); ++k) {
        const auto b = memberships[k].box;
        const bool now = m_covered[i][k];
        const bool then =
            query::intersects(candidate, m_objective.workloads().range.queries[b].box);
        if (now == then)
            continue;
        const auto next = m_found[b] - now + then;
        change += f1(b, next) - f1(b, m_found[b]);
    }
    return m_objective.scale(Objective::Range) * change / static_cast<double>(m_scored);
}

std::vector<char> RangeState::coverage(std::size_t i, const core::TrajectoryView &candidate) const {
    const auto memberships = m_objective.boxes(i);
    std::vector<char> covered;
    covered.reserve(memberships.size());
    for (const auto &membership : memberships)
        covered.push_back(query::intersects(
            candidate, m_objective.workloads().range.queries[membership.box].box));
    return covered;
}

RangeState::Proposal RangeState::propose(std::size_t i, std::vector<char> covered) const {
    Proposal proposal{i, 0.0, std::move(covered)};
    const auto memberships = m_objective.boxes(i);
    if (proposal.covered.size() != memberships.size())
        throw std::invalid_argument("range proposal: one outcome per box needed");
    double change = 0.0;
    for (std::size_t k = 0; k < memberships.size(); ++k) {
        if (m_covered[i][k] == proposal.covered[k])
            continue;
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
    // Each direction from its compressed side
    std::vector<std::vector<char>> outcome(n);
    core::parallel_for(n, [&](std::size_t i) { outcome[i] = outcomes(i, state[i]); });
    for (std::size_t i = 0; i < n; ++i) {
        const auto links = objective.links(i);
        for (std::size_t k = 0; k < links.size(); ++k) {
            m_matched[slot(i, links[k])] = outcome[i][k];
            if (outcome[i][k]) {
                const std::size_t t = count(links[k]);
                ++m_found[t];
                m_hits[t] += links[k].positive;
            }
        }
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

void JoinState::update_slot(const JoinLink &link, long delta, std::span<double> sum,
                            std::span<long> n_counted) const {
    const auto t = count(link);
    const auto term = t / m_objective.db().size();
    const auto old_found = m_found[t], old_hits = m_hits[t];
    const auto new_found = static_cast<std::size_t>(static_cast<long>(old_found) + delta);
    const auto new_hits =
        static_cast<std::size_t>(static_cast<long>(old_hits) + (link.positive ? delta : 0));
    if (counted(t, old_found)) {
        sum[term] -= f1(t, old_hits, old_found);
        --n_counted[term];
    }
    if (counted(t, new_found)) {
        sum[term] += f1(t, new_hits, new_found);
        ++n_counted[term];
    }
}

double JoinState::metric() const { return mean_f1(m_objective, m_sum, m_counted); }

JoinState::Proposal JoinState::propose(std::size_t i, const core::TrajectoryView &candidate) const {
    return propose(i, outcomes(i, candidate));
}

double JoinState::gain(std::size_t i, const core::TrajectoryView &candidate,
                       double &own_error) const {
    const auto links = m_objective.links(i);
    if (links.empty())
        return 0.0;
    std::array<double, kMaxJoinTerms> sum{};
    std::array<long, kMaxJoinTerms> n_counted{};
    std::ranges::copy(m_sum, sum.begin());
    std::ranges::copy(m_counted, n_counted.begin());
    bool changed = false;
    // Each link of i has a distinct (link.other, link.term), hence a distinct query slot t.
    for (const auto &link : links) {
        const bool now = m_matched[slot(i, link)];
        const bool then = m_objective.held_out_match(i, link, candidate, own_error);
        if (now == then)
            continue;
        changed = true;
        update_slot(link, then ? 1 : -1, sum, n_counted);
    }
    if (!changed)
        return 0.0;
    const auto terms = m_objective.join_terms();
    const double after =
        mean_f1(m_objective, std::span(sum).first(terms), std::span(n_counted).first(terms));
    return m_objective.scale(Objective::Join) * (after - metric());
}

std::vector<char> JoinState::outcomes(std::size_t i, const core::TrajectoryView &candidate) const {
    std::vector<char> matched;
    matched.reserve(m_objective.links(i).size());
    double own_error = -1.0;
    for (const auto &link : m_objective.links(i))
        matched.push_back(m_objective.held_out_match(i, link, candidate, own_error));
    return matched;
}

JoinState::Proposal JoinState::propose(std::size_t i, std::vector<char> matched) const {
    Proposal proposal{i, 0.0, std::move(matched)};
    const auto links = m_objective.links(i);
    if (proposal.matched.size() != links.size())
        throw std::invalid_argument("join proposal: outcome count does not match links");
    if (links.empty())
        return proposal;
    std::array<double, kMaxJoinTerms> sum{};
    std::array<long, kMaxJoinTerms> n_counted{};
    std::ranges::copy(m_sum, sum.begin());
    std::ranges::copy(m_counted, n_counted.begin());
    bool changed = false;
    // Each link of i has a distinct (link.other, link.term), hence a distinct query slot t.
    for (std::size_t k = 0; k < links.size(); ++k) {
        const auto &link = links[k];
        const bool now = m_matched[slot(i, link)];
        const bool then = proposal.matched[k];
        if (now == then)
            continue;
        changed = true;
        update_slot(link, then ? 1 : -1, sum, n_counted);
    }
    if (!changed)
        return proposal;
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
        const auto t = count(link);
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

KnnState::KnnState(const Objective &objective, const KeptSets &state, std::size_t watch)
    : m_objective(objective), m_watch(watch) {
    const auto &db = objective.db();
    const auto &workloads = objective.workloads();
    const std::size_t n = db.size();
    m_list.resize(n);
    m_distance.resize(n);
    m_hits.assign(n, 0);
    m_watchers.resize(n);
    const auto compressed = core::materialize(db, state);
    const PreparedTrajectories prepared(db, state);
    const auto views = prepared.views();
    // Each list with its distances in state; the search already has its own neighbours'
    core::parallel_for(n, [&](std::size_t e) {
        const auto &entry = workloads.knn[e];
        if (!entry.valid)
            return;
        auto &list = m_list[e];
        list.assign(entry.top.begin(), entry.top.end());
        constexpr double unknown = -1.0;
        auto &distance = m_distance[e];
        distance.assign(list.size(), unknown);
        const auto query = db.trajectory(e);
        const std::size_t count = std::min(watch + 1, n);
        // Either variant: the search's own distances, as knn_distance gives them
        for (const auto &[j, d] :
             query::knn_with_distances(compressed, query, count, workloads.frechet)) {
            if (j == e)
                continue;
            const auto at = std::ranges::find(list, j);
            if (at == list.end()) {
                list.push_back(static_cast<std::uint32_t>(j));
                distance.push_back(d);
            } else {
                distance[static_cast<std::size_t>(at - list.begin())] = d;
            }
        }
        for (std::size_t p = 0; p < list.size(); ++p)
            if (distance[p] == unknown)
                distance[p] = query::knn_distance(query, views[list[p]], workloads.frechet);
    });
    for (std::size_t e = 0; e < n; ++e) {
        m_valid += workloads.knn[e].valid;
        for (std::size_t p = 0; p < m_list[e].size(); ++p)
            m_watchers[m_list[e][p]].push_back(
                {static_cast<std::uint32_t>(e), static_cast<std::uint32_t>(p)});
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
    const auto &watchers = m_watchers[i];
    std::vector<double> result(watchers.size());
    auto one = [&](std::size_t s) {
        result[s] = query::knn_distance(m_objective.db().trajectory(watchers[s].query), candidate,
                                        m_objective.workloads().frechet);
    };
    if (parallel)
        core::parallel_for(watchers.size(), one);
    else
        for (std::size_t s = 0; s < watchers.size(); ++s)
            one(s);
    return result;
}

double KnnState::value(long change) const {
    return m_valid ? m_objective.scale(Objective::Knn) * static_cast<double>(change) /
                         static_cast<double>(m_objective.workloads().k * m_valid)
                   : 0.0;
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

void KnnState::watch(std::span<const std::vector<core::TrajectoryView>> versions,
                     std::span<const core::TrajectoryView> state) {
    const auto &db = m_objective.db();
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
    // A version keeps i's endpoints, so the squared endpoint bound rules out most queries for all
    // of them without touching TrajectoryDb or taking square roots
    struct Endpoints {
        double x0, y0, x1, y1, radius_sq;
    };
    std::vector<Endpoints> queries(n);
    for (std::size_t e = 0; e < n; ++e) {
        if (radius[e] < 0) {
            queries[e] = {0.0, 0.0, 0.0, 0.0, -1.0};
            continue;
        }
        const auto q = db.trajectory(e);
        const std::size_t last = q.size() - 1;
        queries[e] = {q.x[0], q.y[0], q.x[last], q.y[last], radius[e] * radius[e] * (1.0 + 1e-12)};
    }
    std::vector<std::vector<std::pair<std::uint32_t, double>>> added(n);
    core::parallel_for(n, [&](std::size_t i) {
        if (versions[i].empty())
            return;
        const auto original = db.trajectory(i);
        const std::size_t last = original.size() - 1;
        const double ix0 = original.x[0], iy0 = original.y[0];
        const double ix1 = original.x[last], iy1 = original.y[last];
        for (std::size_t e = 0; e < n; ++e) {
            const auto &q = queries[e];
            if (q.radius_sq < 0 || e == i)
                continue;
            const double dx0 = q.x0 - ix0, dy0 = q.y0 - iy0;
            if (dx0 * dx0 + dy0 * dy0 > q.radius_sq)
                continue;
            const double dx1 = q.x1 - ix1, dy1 = q.y1 - iy1;
            if (dx1 * dx1 + dy1 * dy1 > q.radius_sq)
                continue;
            if (std::ranges::any_of(m_watchers[i], [&](const Watcher &w) { return w.query == e; }))
                continue;
            for (const auto &version : versions[i])
                if (m_objective.original_within(e, version, radius[e])) {
                    added[i].push_back({static_cast<std::uint32_t>(e),
                                        query::knn_distance(db.trajectory(e), state[i],
                                                            m_objective.workloads().frechet)});
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

double KnnState::propose_bounded(std::size_t i, const core::TrajectoryView &candidate, double reach,
                                 double beat) const {
    const auto &watchers = m_watchers[i];
    if (watchers.empty() || m_valid == 0 || m_objective.scale(Objective::Knn) <= 0.0)
        return 0.0;
    const std::size_t k = m_objective.workloads().k;
    const bool continuous = m_objective.workloads().frechet == query::FrechetVariant::Continuous;
    // Clear of the k-th place by more than the stored distances' error
    constexpr double margin = 2.5 * kKnnDistanceError;
    const bool prune = std::isfinite(beat);
    long optimistic = 0;
    if (prune) {
        for (const auto &[e, position] : watchers) {
            const auto inside = hits(e, position, -1.0);
            const auto outside = hits(e, position, std::numeric_limits<double>::infinity());
            optimistic +=
                static_cast<long>(std::max(inside, outside)) - static_cast<long>(m_hits[e]);
        }
        if (value(optimistic) < beat - kBoundSlack)
            return value(optimistic);
    }
    long change = 0;
    for (const auto &[e, position] : watchers) {
        const auto inside = hits(e, position, -1.0);
        const auto outside = hits(e, position, std::numeric_limits<double>::infinity());
        const long cur = static_cast<long>(m_hits[e]);
        std::size_t now = inside;
        if (inside != outside) {
            // The k-th distance among the others, from the ranking (it has k + 1 entries here):
            // the (k + 1)-th overall if i is in the top k, else the k-th. i is in clearly below
            // it, out clearly above
            const auto &rank = m_rank[e];
            const bool in_top =
                std::find(rank.begin(), rank.begin() + static_cast<std::ptrdiff_t>(k),
                          static_cast<std::uint32_t>(position)) !=
                rank.begin() + static_cast<std::ptrdiff_t>(k);
            const double kth = m_distance[e][rank[in_top ? k : k - 1]];
            // |d(query, candidate) - d(query, current)| <= reach for continuous Fréchet; the stored
            // key is the current distance rounded up to the grid
            const double at = m_distance[e][position];
            if (continuous && kth - margin > 0 &&
                at + reach <= (kth - margin) * (1.0 - kCertificateMargin))
                now = inside;
            else if (continuous && at - 2.0 * kKnnDistanceError - reach >
                                       (kth + margin) * (1.0 + kCertificateMargin))
                now = outside;
            else if (kth - margin > 0 && m_objective.original_within(e, candidate, kth - margin))
                now = inside;
            else if (!m_objective.original_within(e, candidate, kth + margin))
                now = outside;
            else
                now = hits(e, position,
                           query::knn_distance(m_objective.db().trajectory(e), candidate,
                                               m_objective.workloads().frechet));
        }
        const long actual = static_cast<long>(now) - cur;
        change += actual;
        if (prune) {
            optimistic += actual - (static_cast<long>(std::max(inside, outside)) - cur);
            if (value(optimistic) < beat - kBoundSlack)
                return value(optimistic);
        }
    }
    return value(change);
}

std::vector<double> KnnState::stored(std::size_t i) const {
    std::vector<double> result;
    result.reserve(m_watchers[i].size());
    for (const auto &watcher : m_watchers[i])
        result.push_back(m_distance[watcher.query][watcher.position]);
    return result;
}

double KnnState::propose(std::size_t i, std::span<const double> distances) const {
    long change = 0;
    for (std::size_t s = 0; s < m_watchers[i].size(); ++s) {
        const auto &watcher = m_watchers[i][s];
        change += static_cast<long>(hits(watcher.query, watcher.position, distances[s])) -
                  static_cast<long>(m_hits[watcher.query]);
    }
    return value(change);
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
    return m_valid ? static_cast<double>(m_hit_sum) /
                         static_cast<double>(m_objective.workloads().k * m_valid)
                   : 0.0;
}

} // namespace teacher
