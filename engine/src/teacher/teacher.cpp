#include "backend/engine/teacher/teacher.hpp"

#include "backend/engine/allocation/budget_allocator.hpp"
#include "backend/engine/core/parallel.hpp"
#include "backend/engine/core/timer.hpp"
#include "backend/engine/selection/rdp.hpp"
#include "backend/engine/selection/sed.hpp"
#include "backend/engine/selection/selector.hpp"
#include "backend/engine/teacher/curves.hpp"
#include "backend/engine/teacher/greedy.hpp"
#include "backend/engine/teacher/trade_matching.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <stdexcept>

namespace teacher {

namespace {

using selection::AlgorithmType;
enum CandidateIndex : std::size_t {
    kRdp,
    kSed,
    kUniform,
    kGreedy,
    kGreedyJoint,
    kCandidateCount,
};

// Classic selectors, then pure query greedy and adaptive joint greedy
constexpr std::array<std::string_view, kCandidateCount> kCandidates{"RDP", "SED", "UNIFORM",
                                                                    "GREEDY", "GREEDY_JOINT"};
constexpr std::size_t kMaxResponsePasses = 10;
constexpr std::size_t kKnnWatchFactor = 3;
constexpr std::size_t kMaxRefineCycles = 3; // 2 (refine -> point-trade) cycles + final refine
constexpr std::size_t kMaxRefineRounds = 8;
constexpr double kMinRoundGain = 1.5e-4;
constexpr double kMinImprovement = 1e-12;

// candidates[c][i]: trajectory i simplified by kCandidates[c] at budgets[i]
using Candidates = std::array<KeptSets, kCandidates.size()>;
using Methods = std::vector<std::size_t>; // per trajectory

// The first candidate up to c that keeps the same points as c, kept(k) giving candidate k's set
template <typename Kept> std::size_t first_equal(std::size_t c, Kept &&kept) {
    std::size_t first = 0;
    while (first < c && kept(first) != kept(c))
        ++first;
    return first;
}

// One trajectory's candidates up to cap, from sequences built once. A candidate depends only on
// (i, c, b, cap), so curves and final kept sets agree
class CandidateMaker {
  public:
    CandidateMaker(const Objective &objective, std::size_t i, std::size_t cap)
        : m_original(objective.db().trajectory(i)), m_cap(std::min(cap, m_original.size())),
          m_objective(objective), m_index(i) {}

    std::vector<std::size_t> make(std::size_t candidate, std::size_t budget) {
        budget = std::min(budget, m_original.size());
        if (budget > m_cap)
            throw std::runtime_error("teacher: candidate budget above its cap");
        if (budget <= 2 || budget == m_original.size())
            return selection::simplify_trajectory(m_original, AlgorithmType::UNIFORM, budget);
        switch (candidate) {
        case kRdp:
            return simplify(AlgorithmType::RDP, budget);
        case kSed:
            return simplify(AlgorithmType::SED, budget);
        case kUniform:
            return simplify(AlgorithmType::UNIFORM, budget);
        case kGreedy:
            return prefix(greedy_sequence(), budget);
        case kGreedyJoint:
            return prefix(joint_sequence(), budget);
        default:
            throw std::logic_error("teacher: unknown candidate");
        }
    }

    const Greedy &greedy() {
        if (!m_greedy)
            m_greedy.emplace(m_objective, m_index);
        return *m_greedy;
    }

  private:
    const std::vector<std::size_t> &sequence_of(AlgorithmType type) {
        auto &cached = type == AlgorithmType::RDP ? m_rdp : m_sed;
        if (!cached)
            cached = type == AlgorithmType::RDP ? selection::rdp::sequence(m_original, m_cap)
                                                : selection::sed::sequence(m_original, m_cap);
        return *cached;
    }
    std::vector<std::size_t> simplify(AlgorithmType type, std::size_t b) {
        if (type == AlgorithmType::UNIFORM)
            return selection::simplify_trajectory(m_original, type, b);
        return prefix(sequence_of(type), b);
    }
    static std::vector<std::size_t> prefix(const std::vector<std::size_t> &sequence,
                                           std::size_t b) {
        std::vector<std::size_t> kept(sequence.begin(),
                                      sequence.begin() + static_cast<std::ptrdiff_t>(b));
        std::ranges::sort(kept);
        return kept;
    }
    const std::vector<std::size_t> &greedy_sequence() {
        if (!m_greedy)
            m_greedy.emplace(m_objective, m_index);
        if (!m_greedy_seq)
            m_greedy_seq = m_greedy->sequence(m_cap, sequence_of(AlgorithmType::SED));
        return *m_greedy_seq;
    }

    const std::vector<std::size_t> &joint_sequence() {
        if (!m_greedy)
            m_greedy.emplace(m_objective, m_index);
        if (!m_joint)
            m_joint =
                m_greedy->joint_sequence(m_cap, m_objective, m_index, m_objective.segments(m_index),
                                         sequence_of(AlgorithmType::RDP));
        return *m_joint;
    }

    core::TrajectoryView m_original;
    std::size_t m_cap;
    const Objective &m_objective;
    std::size_t m_index;
    std::optional<Greedy> m_greedy;
    std::optional<std::vector<std::size_t>> m_rdp, m_sed, m_greedy_seq, m_joint;
};

// Per trajectory, built on first use and kept: its sequences serve the curves, the candidates and
// the trades alike
using Makers = std::vector<std::optional<CandidateMaker>>;

CandidateMaker &maker_for(Makers &makers, const Objective &objective, std::size_t i,
                          std::size_t cap) {
    if (!makers[i])
        makers[i].emplace(objective, i, cap);
    return *makers[i];
}

// All candidates at each trajectory's budget, or only the selected one
Candidates make_candidates(const Objective &objective, const std::vector<std::size_t> &budgets,
                           const std::vector<std::size_t> &caps, Makers &makers,
                           const Methods *selected = nullptr) {
    Candidates candidates;
    for (auto &kept : candidates)
        kept.resize(budgets.size());
    core::parallel_for(budgets.size(), [&](std::size_t i) {
        auto &maker = maker_for(makers, objective, i, caps[i]);
        if (selected) {
            const auto c = (*selected)[i];
            candidates[c][i] = maker.make(c, budgets[i]);
        } else {
            for (std::size_t c = 0; c < kCandidates.size(); ++c)
                candidates[c][i] = maker.make(c, budgets[i]);
        }
    });
    return candidates;
}

KeptSets assemble(const Candidates &candidates, const Methods &method) {
    KeptSets kept(method.size());
    for (std::size_t i = 0; i < method.size(); ++i)
        kept[i] = candidates[method[i]][i];
    return kept;
}

struct Pick {
    std::size_t method;
    double value, rdp;
};

// Best candidate by independent_value, ties to the lower index. RDP is scored exactly, the rest
// best bound first so losers skip most kNN checks
Pick pick(const Objective &objective, std::size_t i,
          std::span<const std::pair<std::size_t, core::TrajectoryView>> options,
          std::span<const core::TrajectoryView> partners, std::span<const double> range_weights,
          const JoinState &join) {
    const double rdp =
        objective.independent_value(i, options[0].second, partners,
                                    -std::numeric_limits<double>::infinity(), range_weights, &join);
    Pick best{options[0].first, rdp, rdp};
    std::vector<Objective::IndependentScore> scores;
    scores.reserve(options.size() - 1);
    std::vector<std::pair<double, std::size_t>> order; // bound, position in options
    order.reserve(options.size() - 1);
    for (std::size_t k = 1; k < options.size(); ++k) {
        scores.push_back(
            objective.prepare_independent(i, options[k].second, partners, range_weights, &join));
        order.push_back({scores.back().bound, k});
    }
    std::ranges::sort(order, std::greater{});
    for (const auto &[bound, k] : order) {
        if (bound < best.value - kBoundSlack)
            break;
        const auto c = options[k].first;
        const double value = objective.independent_value(scores[k - 1], best.value);
        if (value > best.value || (value == best.value && c < best.method)) {
            best.value = value;
            best.method = c;
        }
    }
    return best;
}

// Value curves up to caps[i], best candidate and RDP alone, partners fixed
struct Curves {
    std::vector<Curve> best, rdp;
};

Curves build_curves(const core::TrajectoryDb &db, const Objective &objective,
                    const KeptSets &partners, const std::vector<std::size_t> &caps, Makers &makers,
                    bool rdp_only) {
    Curves curves{std::vector<Curve>(db.size()), std::vector<Curve>(db.size())};
    const PreparedTrajectories prepared(db, partners);
    const auto partner_views = prepared.views();
    const RangeState range(objective, partner_views);
    const JoinState join(objective, partner_views);
    core::parallel_for(db.size(), [&](std::size_t i) {
        const auto range_weights = range.weights(i);
        auto &maker = maker_for(makers, objective, i, caps[i]);
        const std::size_t max_b = std::min(caps[i], db.trajectory(i).size());
        auto &best = curves.best[i];
        auto &rdp = curves.rdp[i];
        best.value.reserve(max_b - 1);
        best.method.reserve(max_b - 1);
        rdp.value.reserve(max_b - 1);
        rdp.method.reserve(max_b - 1);
        std::array<std::vector<std::size_t>, kCandidates.size()> kept;
        std::vector<PreparedTrajectory> owned;
        std::vector<std::pair<std::size_t, core::TrajectoryView>> options;
        owned.reserve(kCandidates.size());
        options.reserve(kCandidates.size());
        for (std::size_t b = 2; b <= max_b; ++b) {
            options.clear();
            for (std::size_t c = 0; c < (rdp_only ? 1 : kCandidates.size()); ++c) {
                kept[c] = maker.make(c, b);
                if (first_equal(c, [&](std::size_t k) -> const auto & { return kept[k]; }) != c)
                    continue;
                const auto slot = options.size();
                if (slot == owned.size())
                    owned.emplace_back(db.trajectory(i), kept[c]);
                else
                    owned[slot].assign(db.trajectory(i), kept[c]);
                options.push_back({c, owned[slot].view()});
            }
            const auto chosen = pick(objective, i, options, partner_views, range_weights, join);
            rdp.value.push_back(chosen.rdp);
            rdp.method.push_back(kRdp);
            best.value.push_back(chosen.value);
            best.method.push_back(static_cast<std::uint8_t>(chosen.method));
        }
    });
    return curves;
}

Methods curve_methods(const std::vector<Curve> &curves, const std::vector<std::size_t> &budgets) {
    Methods method(curves.size());
    for (std::size_t i = 0; i < curves.size(); ++i) {
        if (budgets[i] < 2 || budgets[i] > curves[i].max_budget())
            throw std::runtime_error("teacher: budget outside its value curve");
        method[i] = curves[i].method[budgets[i] - 2];
    }
    return method;
}

// The loop: coordinate ascent on the tracked J_T, one trajectory at a time in random order
class Search {
  public:
    Search(const core::TrajectoryDb &db, const Objective &objective, const Candidates &candidates,
           const Methods &method)
        : m_start(method) {
        auto start = std::chrono::steady_clock::now();
        // Every distinct candidate prepared, since any of them can enter a query's watch list
        m_prepared.resize(db.size());
        for (auto &views : m_views)
            views.resize(db.size());
        m_representative.resize(db.size());
        core::parallel_for(db.size(), [&](std::size_t i) {
            auto &owned = m_prepared[i];
            // Reserve first: views borrow from these
            owned.reserve(kCandidates.size());
            for (std::size_t c = 0; c < kCandidates.size(); ++c) {
                const auto first =
                    first_equal(c, [&](std::size_t k) -> const auto & { return candidates[k][i]; });
                m_representative[i][c] = first;
                if (first == c) {
                    owned.emplace_back(db.trajectory(i), candidates[c][i]);
                    m_views[c][i] = owned.back().view();
                } else {
                    m_views[c][i] = m_views[first][i];
                }
            }
        });
        m_state.resize(db.size());
        for (std::size_t i = 0; i < db.size(); ++i)
            m_state[i] = m_views[method[i]][i];
        const double prepare_s = core::seconds_since(start);
        start = std::chrono::steady_clock::now();
        // Lists from the start, then every query also watches any candidate within the distance of
        // its watch-th nearest there
        m_knn.emplace(objective, assemble(candidates, method),
                      kKnnWatchFactor * objective.workloads().k);
        std::vector<std::vector<core::TrajectoryView>> versions(db.size());
        for (std::size_t i = 0; i < db.size(); ++i)
            for (std::size_t c = 0; c < kCandidates.size(); ++c)
                if (m_representative[i][c] == c)
                    versions[i].push_back(m_views[c][i]);
        m_knn->watch(versions, m_state);
        const double knn_s = core::seconds_since(start);
        start = std::chrono::steady_clock::now();
        const auto active = [&](std::size_t i) {
            return !objective.boxes(i).empty() || !objective.links(i).empty() ||
                   !objective.probes(i).empty() || m_knn->watched(i) ||
                   objective.scale(Objective::Shape) > 0 || objective.scale(Objective::Margin) > 0;
        };
        for (std::size_t i = 0; i < db.size(); ++i)
            if (active(i))
                m_order.push_back(i);
        std::cout << std::format("teacher:   loop setup: prepare {:.1f} s\n", prepare_s);
        m_join.emplace(objective, m_state);
        m_range.emplace(objective, m_state);
        // Every candidate's deviation, separable value, box coverage and join outcomes; none
        // depend on the others. kNN is scored on the fly with propose_bounded and distances are
        // computed only for trajectories that switch
        m_deviation.resize(db.size());
        m_separable.resize(db.size());
        m_coverage.resize(db.size());
        m_outcomes.resize(db.size());
        core::parallel_for(db.size(), [&](std::size_t i) {
            if (!active(i))
                return;
            for (std::size_t c = 0; c < kCandidates.size(); ++c) {
                const auto first = m_representative[i][c];
                if (first != c) {
                    m_deviation[i][c] = m_deviation[i][first];
                    m_separable[i][c] = m_separable[i][first];
                    continue;
                }
                const auto prof = objective.deviation_profile(i, m_views[c][i]);
                m_deviation[i][c] = prof.max_m;
                m_separable[i][c] = objective.separable(i, m_views[c][i], prof);
                m_coverage[i][c] = m_range->coverage(i, m_views[c][i]);
                m_outcomes[i][c] = m_join->outcomes(i, m_views[c][i]);
            }
        });
        std::cout << std::format("teacher:   loop setup: kNN lists {:.1f} s, candidates {:.1f} s\n",
                                 knn_s, core::seconds_since(start));
    }
    Search(const Search &) = delete;
    Search &operator=(const Search &) = delete;

    struct Trackers {
        KnnState &knn;
        JoinState &join;
        RangeState &range;
    };
    Trackers trackers() { return {*m_knn, *m_join, *m_range}; }

    // Once: the trackers are updated in place
    Methods run(std::uint64_t seed) {
        auto method = m_start;
        auto &join = *m_join;
        auto &range = *m_range;
        auto &knn = *m_knn;
        std::mt19937_64 rng(seed + kOrderSeedOffset);

        for (std::size_t pass = 1; pass <= kMaxResponsePasses; ++pass) {
            const auto start = std::chrono::steady_clock::now();
            std::ranges::shuffle(m_order, rng);
            std::size_t switches = 0;
            for (const auto i : m_order) {
                const double here = m_separable[i][method[i]];
                const double cur_dev = m_deviation[i][method[i]];
                std::size_t best = method[i];
                JoinState::Proposal best_proposal;
                RangeState::Proposal best_range;
                double best_gain = kMinImprovement; // strict gains only, so ties cannot cycle
                for (std::size_t c = 0; c < kCandidates.size(); ++c) {
                    // Each distinct kept set is tried once, and never the one i already has
                    const bool duplicate = m_representative[i][c] != c;
                    const bool current = m_representative[i][c] == m_representative[i][method[i]];
                    if (duplicate || current)
                        continue;
                    auto proposal = join.propose(i, m_outcomes[i][c]);
                    auto range_proposal = range.propose(i, m_coverage[i][c]);
                    const double base =
                        m_separable[i][c] - here + proposal.gain + range_proposal.gain;
                    const double gain =
                        base + knn.propose_bounded(i, m_views[c][i], cur_dev + m_deviation[i][c],
                                                   best_gain - base);
                    if (gain > best_gain) {
                        best = c;
                        best_gain = gain;
                        best_proposal = std::move(proposal);
                        best_range = std::move(range_proposal);
                    }
                }
                if (best == method[i])
                    continue;
                method[i] = best;
                ++switches;
                m_state[i] = m_views[best][i];
                join.apply(best_proposal);
                range.apply(best_range);
                knn.apply(i, knn.distances(i, m_views[best][i], true));
            }
            std::cout << std::format("teacher:   pass {}: {} switches, join F1 {:.4f}, knn {:.4f}, "
                                     "{:.1f} s\n",
                                     pass, switches, join.metric(), knn.metric(),
                                     core::seconds_since(start));
            if (switches == 0)
                break;
        }
        return method;
    }

  private:
    Methods m_start;
    std::vector<std::size_t> m_order;
    std::vector<std::vector<PreparedTrajectory>> m_prepared;
    std::array<std::vector<core::TrajectoryView>, kCandidates.size()> m_views;
    std::vector<std::array<double, kCandidates.size()>> m_deviation, m_separable;
    std::vector<std::array<std::size_t, kCandidates.size()>> m_representative;
    std::vector<std::array<std::vector<char>, kCandidates.size()>> m_coverage, m_outcomes;
    std::optional<KnnState> m_knn;
    std::vector<core::TrajectoryView> m_state;
    std::optional<JoinState> m_join;
    std::optional<RangeState> m_range;
};

// The loop accepts switches on the tracked J_T, whose kNN term only watches each query's top k
// plus its nearest others in the start state and the candidates. In a mixed state an unwatched
// trajectory can enter a top k unseen, so the result is checked exactly against the start.
// before is the start's exact J_T when the caller already has it
struct Checked {
    Methods methods;
    double value; // exact J_T of methods
    Objective::Terms metrics;
    bool reverted = false;
};
Checked checked_loop(const Objective &objective, const Candidates &candidates, const Methods &start,
                     std::optional<Objective::Terms> before_metrics, Methods looped,
                     std::string_view name) {
    if (!before_metrics)
        before_metrics = objective.metrics(assemble(candidates, start));
    const double before = objective.total(*before_metrics);
    auto after_metrics = objective.metrics(assemble(candidates, looped));
    const double after = objective.total(after_metrics);
    const bool lost = after < before;
    std::cout << std::format("teacher: {} exact check, J_T {:.6f} -> {:.6f}{}\n", name, before,
                             after, lost ? ", lost: keeping the start" : "");
    if (lost)
        return {start, before, *before_metrics, true};
    return {std::move(looped), after, after_metrics, false};
}

struct Refined {
    KeptSets kept;
    double value;
    std::optional<Objective::Terms> metrics;
    bool reverted = false;
};

// Alternating free-knot refinement (knot shifts at fixed per-trajectory budgets) and 1-point
// cross-trajectory budget trading against live tracked exact J_T
Refined refine_points(const core::TrajectoryDb &db, const Objective &objective,
                      const std::vector<std::size_t> &caps, Makers &makers, KeptSets kept,
                      double start_value, Search::Trackers *reused = nullptr) {
    const std::size_t n = db.size();
    std::vector<PreparedTrajectory> owned;
    owned.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        owned.emplace_back(db.trajectory(i), kept[i]);
    std::vector<core::TrajectoryView> state(n);
    for (std::size_t i = 0; i < n; ++i)
        state[i] = owned[i].view();
    std::optional<JoinState> owned_join;
    std::optional<RangeState> owned_range;
    std::optional<KnnState> owned_knn;
    if (!reused) {
        owned_join.emplace(objective, state);
        owned_range.emplace(objective, state);
        owned_knn.emplace(objective, kept, kKnnWatchFactor * objective.workloads().k);
    }
    JoinState &join = reused ? reused->join : *owned_join;
    RangeState &range = reused ? reused->range : *owned_range;
    KnnState &knn = reused ? reused->knn : *owned_knn;
    std::vector<double> separable(n), state_error(n);
    core::parallel_for(n, [&](std::size_t i) {
        const auto prof = objective.deviation_profile(i, state[i]);
        state_error[i] = prof.max_m;
        separable[i] = objective.separable(i, state[i], prof);
    });

    struct Offer {
        double gain = kMinImprovement;
        std::vector<std::size_t> kept;
    };
    const auto start_kept = kept;
    double value = start_value;
    std::cout << std::format("teacher: refining from J_T {:.6f}\n", value);
    auto run_refine_rounds = [&](std::size_t max_rounds, double min_round_gain,
                                 std::string_view label) {
        for (std::size_t round = 1; round <= max_rounds; ++round) {
            auto start = std::chrono::steady_clock::now();
            std::vector<Offer> offers(n);
            core::parallel_for(n, [&](std::size_t i) {
                if (kept[i].size() < 3 || kept[i].size() >= db.trajectory(i).size())
                    return;
                auto &maker = maker_for(makers, objective, i, caps[i]);
                const auto rw = range.weights(i);
                const auto proposals = maker.greedy().refine_candidates(
                    kept[i], rw, objective.scale(Objective::Range), objective.segments(i));
                Offer best;
                for (const auto &candidate : proposals) {
                    const PreparedTrajectory prepared(db.trajectory(i), candidate);
                    const auto view = prepared.view();
                    const auto prof = objective.deviation_profile(i, view);
                    double dev = prof.max_m;
                    const double base = objective.separable(i, view, prof) - separable[i] +
                                        join.gain(i, view, dev) + range.gain(i, view);
                    const double gain =
                        base + knn.propose_bounded(i, view, state_error[i] + dev, best.gain - base);
                    if (gain > best.gain)
                        best = {gain, candidate};
                }
                if (!best.kept.empty())
                    offers[i] = std::move(best);
            });
            const double sweep_s = core::seconds_since(start);
            start = std::chrono::steady_clock::now();

            std::vector<std::pair<double, std::size_t>> active;
            for (std::size_t i = 0; i < n; ++i)
                if (!offers[i].kept.empty())
                    active.push_back({offers[i].gain, i});
            if (active.empty()) {
                std::cout << std::format("teacher: {} round {}: 0 candidates, sweep {:.1f} s\n",
                                         label, round, sweep_s);
                break;
            }
            const double before_watch = knn.metric();
            {
                std::vector<std::vector<PreparedTrajectory>> prepared(n);
                std::vector<std::vector<core::TrajectoryView>> versions(n);
                for (const auto &[gain, i] : active) {
                    prepared[i].emplace_back(db.trajectory(i), offers[i].kept);
                    versions[i].push_back(prepared[i].back().view());
                }
                knn.watch(versions, state);
            }
            const double knn_correction =
                objective.scale(Objective::Knn) * (knn.metric() - before_watch);
            value += knn_correction;
            const double watch_s = core::seconds_since(start);
            start = std::chrono::steady_clock::now();

            std::ranges::sort(active, [](const auto &a, const auto &b) {
                return a.first > b.first || (a.first == b.first && a.second < b.second);
            });
            // Every active candidate's deviation, separable score, join outcomes, range coverage,
            // and watched kNN distances depend only on that trajectory and fixed originals, so
            // compute all of them in parallel before the sequential exact commit loop
            const std::size_t m = active.size();
            std::vector<std::optional<PreparedTrajectory>> next_prepared(m);
            std::vector<double> next_dev(m), next_sep(m);
            std::vector<std::vector<char>> next_join(m), next_range(m);
            std::vector<std::vector<double>> next_knn(m);
            core::parallel_for(m, [&](std::size_t idx) {
                const auto i = active[idx].second;
                next_prepared[idx].emplace(db.trajectory(i), offers[i].kept);
                const auto view = next_prepared[idx]->view();
                const auto prof = objective.deviation_profile(i, view);
                next_dev[idx] = prof.max_m;
                next_sep[idx] = objective.separable(i, view, prof);
                next_join[idx] = join.outcomes(i, view);
                next_range[idx] = range.coverage(i, view);
                next_knn[idx] = knn.distances(i, view, false);
            });
            const double before = value;
            std::size_t accepted = 0;
            for (std::size_t idx = 0; idx < m; ++idx) {
                const auto i = active[idx].second;
                auto join_prop = join.propose(i, std::move(next_join[idx]));
                auto range_prop = range.propose(i, std::move(next_range[idx]));
                const double gain = next_sep[idx] - separable[i] + join_prop.gain +
                                    range_prop.gain + knn.propose(i, next_knn[idx]);
                if (gain > kMinImprovement) {
                    join.apply(join_prop);
                    range.apply(range_prop);
                    knn.apply(i, next_knn[idx]);
                    owned[i] = std::move(*next_prepared[idx]);
                    state[i] = owned[i].view();
                    kept[i] = std::move(offers[i].kept);
                    separable[i] = next_sep[idx];
                    state_error[i] = next_dev[idx];
                    value += gain;
                    ++accepted;
                }
            }
            std::cout << std::format(
                "teacher: {} round {}: {} candidates, {} kept, J_T {:.6f} -> {:.6f}, "
                "kNN list correction {:+.6f}, sweep {:.1f} s, watch {:.1f} s, commit {:.1f} s\n",
                label, round, active.size(), accepted, before, value, knn_correction, sweep_s,
                watch_s, core::seconds_since(start));
            if (accepted == 0 || (value - before) < min_round_gain)
                break;
        }
    };

    // Point-level budget trading: move 1 point directly from a refined seller (drop 1 knot) to a
    // refined buyer (insert 1 unkept point), breaking the fixed-recipe wall on budget allocation
    // too
    auto run_point_trade_rounds = [&](std::size_t max_rounds, double min_round_gain,
                                      std::string_view label) -> std::size_t {
        std::size_t total_accepted = 0;
        for (std::size_t round = 1; round <= max_rounds; ++round) {
            auto start = std::chrono::steady_clock::now();
            struct TradeOffer {
                double gain = -std::numeric_limits<double>::infinity();
                std::vector<std::size_t> kept;
            };
            std::vector<TradeOffer> buy(n), sell(n);
            core::parallel_for(n, [&](std::size_t i) {
                auto &maker = maker_for(makers, objective, i, caps[i]);
                const auto rw = range.weights(i);
                const auto proposals = maker.greedy().point_trade_candidates(
                    kept[i], rw, objective.scale(Objective::Range), objective.segments(i));
                auto score_best = [&](const std::vector<std::vector<std::size_t>> &cands) {
                    TradeOffer best;
                    for (const auto &candidate : cands) {
                        const PreparedTrajectory prepared(db.trajectory(i), candidate);
                        const auto view = prepared.view();
                        const auto prof = objective.deviation_profile(i, view);
                        double dev = prof.max_m;
                        const double base = objective.separable(i, view, prof) - separable[i] +
                                            join.gain(i, view, dev) + range.gain(i, view);
                        const double need = std::isfinite(best.gain)
                                                ? best.gain - base
                                                : -std::numeric_limits<double>::infinity();
                        const double gain =
                            base + knn.propose_bounded(i, view, state_error[i] + dev, need);
                        if (gain > best.gain)
                            best = {gain, candidate};
                    }
                    return best;
                };
                if (!proposals.adds.empty())
                    buy[i] = score_best(proposals.adds);
                if (!proposals.drops.empty())
                    sell[i] = score_best(proposals.drops);
            });
            const double sweep_s = core::seconds_since(start);
            start = std::chrono::steady_clock::now();

            std::vector<std::pair<double, std::size_t>> buyers, sellers;
            for (std::size_t i = 0; i < n; ++i) {
                if (std::isfinite(buy[i].gain))
                    buyers.push_back({buy[i].gain, i});
                if (std::isfinite(sell[i].gain))
                    sellers.push_back({sell[i].gain, i});
            }
            auto desc_det = [](const auto &a, const auto &b) {
                return a.first > b.first || (a.first == b.first && a.second < b.second);
            };
            std::ranges::sort(buyers, desc_det);
            std::ranges::sort(sellers, desc_det);
            std::vector<char> used(n, 0);
            std::vector<std::pair<std::size_t, std::size_t>> pairs; // (seller, buyer)
            std::size_t s = 0;
            for (const auto &[buy_gain, r] : buyers) {
                if (used[r])
                    continue;
                const auto match =
                    detail::find_profitable_seller(sellers, used, s, r, buy_gain, kMinImprovement);
                if (match == sellers.size())
                    continue;
                const auto seller = sellers[match].second;
                used[r] = used[seller] = 1;
                pairs.push_back({seller, r});
            }
            if (pairs.empty()) {
                std::cout << std::format("teacher: {} round {}: 0 pairs, sweep {:.1f} s\n", label,
                                         round, sweep_s);
                break;
            }

            const double before_watch = knn.metric();
            {
                std::vector<std::vector<PreparedTrajectory>> prepared(n);
                std::vector<std::vector<core::TrajectoryView>> versions(n);
                for (const auto &[seller, buyer] : pairs) {
                    prepared[seller].emplace_back(db.trajectory(seller), sell[seller].kept);
                    versions[seller].push_back(prepared[seller].back().view());
                    prepared[buyer].emplace_back(db.trajectory(buyer), buy[buyer].kept);
                    versions[buyer].push_back(prepared[buyer].back().view());
                }
                knn.watch(versions, state);
            }
            const double knn_correction =
                objective.scale(Objective::Knn) * (knn.metric() - before_watch);
            value += knn_correction;
            const double watch_s = core::seconds_since(start);
            start = std::chrono::steady_clock::now();

            // Precompute exact independent outcomes/distances for all matched sellers and buyers
            const std::size_t pcount = pairs.size();
            std::vector<std::optional<PreparedTrajectory>> s_prep(pcount), b_prep(pcount);
            std::vector<double> s_dev(pcount), s_sep(pcount), b_dev(pcount), b_sep(pcount);
            std::vector<std::vector<char>> s_join(pcount), s_range(pcount), b_join(pcount),
                b_range(pcount);
            std::vector<std::vector<double>> s_knn(pcount), b_knn(pcount);
            core::parallel_for(pcount, [&](std::size_t idx) {
                const auto [seller, buyer] = pairs[idx];
                s_prep[idx].emplace(db.trajectory(seller), sell[seller].kept);
                const auto sv = s_prep[idx]->view();
                const auto s_prof = objective.deviation_profile(seller, sv);
                s_dev[idx] = s_prof.max_m;
                s_sep[idx] = objective.separable(seller, sv, s_prof);
                s_join[idx] = join.outcomes(seller, sv);
                s_range[idx] = range.coverage(seller, sv);
                s_knn[idx] = knn.distances(seller, sv, false);

                b_prep[idx].emplace(db.trajectory(buyer), buy[buyer].kept);
                const auto bv = b_prep[idx]->view();
                const auto b_prof = objective.deviation_profile(buyer, bv);
                b_dev[idx] = b_prof.max_m;
                b_sep[idx] = objective.separable(buyer, bv, b_prof);
                b_join[idx] = join.outcomes(buyer, bv);
                b_range[idx] = range.coverage(buyer, bv);
                b_knn[idx] = knn.distances(buyer, bv, false);
            });

            const double before = value;
            std::size_t accepted = 0;
            for (std::size_t idx = 0; idx < pcount; ++idx) {
                const auto [seller, buyer] = pairs[idx];
                auto old_kept = kept[seller];
                auto old_knn = knn.stored(seller);
                const double old_sep = separable[seller];
                const double old_err = state_error[seller];

                auto s_jp = join.propose(seller, std::move(s_join[idx]));
                auto s_rp = range.propose(seller, std::move(s_range[idx]));
                const double first_gain = s_sep[idx] - separable[seller] + s_jp.gain + s_rp.gain +
                                          knn.propose(seller, s_knn[idx]);
                join.apply(s_jp);
                range.apply(s_rp);
                knn.apply(seller, s_knn[idx]);
                owned[seller] = std::move(*s_prep[idx]);
                state[seller] = owned[seller].view();
                kept[seller] = std::move(sell[seller].kept);
                separable[seller] = s_sep[idx];
                state_error[seller] = s_dev[idx];

                auto b_jp = join.propose(buyer, std::move(b_join[idx]));
                auto b_rp = range.propose(buyer, std::move(b_range[idx]));
                const double second_gain = b_sep[idx] - separable[buyer] + b_jp.gain + b_rp.gain +
                                           knn.propose(buyer, b_knn[idx]);
                if (first_gain + second_gain > kMinImprovement) {
                    join.apply(b_jp);
                    range.apply(b_rp);
                    knn.apply(buyer, b_knn[idx]);
                    owned[buyer] = std::move(*b_prep[idx]);
                    state[buyer] = owned[buyer].view();
                    kept[buyer] = std::move(buy[buyer].kept);
                    separable[buyer] = b_sep[idx];
                    state_error[buyer] = b_dev[idx];
                    value += first_gain + second_gain;
                    ++accepted;
                } else {
                    PreparedTrajectory old_prepared(db.trajectory(seller), old_kept);
                    const auto old_view = old_prepared.view();
                    join.apply(join.propose(seller, old_view));
                    range.apply(range.propose(seller, old_view));
                    knn.apply(seller, old_knn);
                    owned[seller] = std::move(old_prepared);
                    state[seller] = owned[seller].view();
                    kept[seller] = std::move(old_kept);
                    separable[seller] = old_sep;
                    state_error[seller] = old_err;
                }
            }
            total_accepted += accepted;
            std::cout << std::format(
                "teacher: {} round {}: {} pairs, {} kept, J_T {:.6f} -> {:.6f}, "
                "kNN list correction {:+.6f}, sweep {:.1f} s, watch {:.1f} s, commit {:.1f} s\n",
                label, round, pcount, accepted, before, value, knn_correction, sweep_s, watch_s,
                core::seconds_since(start));
            if (accepted == 0 || (value - before) < min_round_gain)
                break;
        }
        return total_accepted;
    };

    // Alternating (refine -> point-trade) cycles + final knot polish under one convergence rule
    for (std::size_t cycle = 1; cycle <= kMaxRefineCycles; ++cycle) {
        run_refine_rounds(kMaxRefineRounds, kMinRoundGain, std::format("refine{}", cycle));
        if (cycle == kMaxRefineCycles ||
            run_point_trade_rounds(kMaxRefineRounds, kMinRoundGain,
                                   std::format("point-trade{}", cycle)) == 0)
            break;
    }
    const auto terms = objective.metrics(kept);
    const double exact = objective.total(terms);
    std::cout << std::format(
        "teacher: refine J_T tracked {:.6f}, exact {:.6f}; tracked/exact kNN {:.6f}/{:.6f}, "
        "range {:.6f}/{:.6f}, join {:.6f}/{:.6f}\n",
        value, exact, knn.metric(), terms[Objective::Knn], range.metric(), terms[Objective::Range],
        join.metric(), terms[Objective::Join]);
    if (exact < start_value)
        return {start_kept, start_value, {}, true};
    return {std::move(kept), exact, terms, false};
}

std::string wins(const Methods &method) {
    std::array<std::size_t, kCandidates.size()> count{};
    for (const auto candidate : method)
        ++count[candidate];
    std::string out;
    for (std::size_t c = 0; c < kCandidates.size(); ++c)
        out += std::format("{}{} {}", c ? ", " : "", kCandidates[c], count[c]);
    return out;
}

std::vector<Arm> collect(const std::map<std::string, Arm, std::less<>> &outputs,
                         std::span<const std::string> arms) {
    std::vector<Arm> result;
    result.reserve(arms.size());
    for (const auto &name : arms)
        result.push_back(outputs.at(name));
    return result;
}

} // namespace

bool is_arm(std::string_view name) { return std::ranges::find(kArms, name) != kArms.end(); }

void check_arms(std::span<const std::string> arms) {
    for (auto a = arms.begin(); a != arms.end(); ++a) {
        if (!is_arm(*a))
            throw std::invalid_argument("unknown teacher arm '" + *a + "'");
        if (std::find(arms.begin(), a, *a) != a)
            throw std::invalid_argument("teacher arm '" + *a + "' listed twice");
    }
}

Teacher::Teacher(const core::TrajectoryDb &db, TeacherWorkloads workloads,
                 const ObjectiveConfig &config, std::uint64_t seed)
    : m_db(db), m_workloads(std::move(workloads)), m_objective(db, m_workloads, config),
      m_seed(seed) {
    const auto &floors = m_objective.floors();
    std::cout << std::format("teacher: {} join pairs ({} positive), floors knn {:.4f}, range "
                             "{:.4f}, where-at {:.4f}, join {:.4f}, shape {:.4f}, margin {:.4f}\n",
                             m_objective.pair_count(), m_objective.positive_count(),
                             floors[Objective::Knn], floors[Objective::Range],
                             floors[Objective::WhereAt], floors[Objective::Join],
                             floors[Objective::Shape], floors[Objective::Margin]);
}

std::vector<Arm> Teacher::run(std::size_t total_budget, std::span<const std::string> arms) const {
    check_arms(arms);
    const auto requested = [&](std::string_view name) {
        return std::ranges::find(arms, name) != arms.end();
    };
    const bool want_budget_rdp = requested("BUDGET_RDP");
    const bool want_choice = requested("BUDGET_CHOICE");
    const bool want_loop = requested("BUDGET_LOOP");
    const bool want_refine = requested("BUDGET_LOOP_REFINE");
    const bool needs_loop = want_loop || want_refine;
    const bool needs_candidates = want_choice || needs_loop;
    if (arms.empty())
        return {};

    const auto &db = m_db;
    const auto &objective = m_objective;
    const auto equal_budgets = allocation::create_trajectory_budgets(
        db, allocation::AllocatorType::SizeProportional, total_budget, 0.0);
    // About 4x the mean interior budget, at least 6, never below the equal share
    const double mean_interior =
        static_cast<double>(total_budget - 2 * db.size()) / static_cast<double>(db.size());
    const auto cap = 2 + static_cast<std::size_t>(std::ceil(std::max(6.0, 4.0 * mean_interior)));
    std::vector<std::size_t> caps(db.size());
    for (std::size_t i = 0; i < db.size(); ++i)
        caps[i] = std::min(std::max(cap, equal_budgets[i]), db.trajectory(i).size());

    std::map<std::string, Arm, std::less<>> outputs;
    // Returns the arm's exact terms; known skips recomputing them
    auto keep = [&](std::string_view name, const Candidates &from, const Methods &method,
                    std::optional<Objective::Terms> known = {}) {
        auto kept = assemble(from, method);
        const auto metrics = known ? *known : objective.metrics(kept);
        const double j_t = objective.total(metrics);
        std::cout << std::format("teacher {}: {}, J_T {:.4f}\n", name, wins(method), j_t);
        outputs[std::string(name)] = {std::string(name), core::to_simplification(kept), j_t,
                                      metrics};
        return metrics;
    };
    auto start = std::chrono::steady_clock::now();
    auto timed = [&](std::string_view step) {
        std::cout << std::format("teacher: {} {:.1f} s\n", step, core::seconds_since(start));
        start = std::chrono::steady_clock::now();
    };
    auto report = [&](std::string_view what, double value,
                      const std::vector<std::size_t> &budgets) {
        std::size_t at_cap = 0;
        for (std::size_t i = 0; i < budgets.size(); ++i)
            at_cap += budgets[i] == caps[i] && caps[i] < db.trajectory(i).size();
        std::cout << std::format("teacher: {}, curve sum {:.4f}, {} trajectories at the cap\n",
                                 what, value, at_cap);
    };

    // Value curves with partners fixed at RDP with equal budgets, so trajectories score
    // independently
    const Methods rdp_methods(db.size(), kRdp);
    Makers makers(db.size());
    const auto partners = make_candidates(objective, equal_budgets, caps, makers, &rdp_methods);
    timed("candidates");
    const auto curves =
        build_curves(db, objective, partners[kRdp], caps, makers, !needs_candidates);
    timed(std::format("value curves up to {} points", cap));

    if (want_budget_rdp) {
        double value = 0;
        const auto budgets = allocate(curves.rdp, total_budget, &value);
        report("RDP allocation", value, budgets);
        keep("BUDGET_RDP", make_candidates(objective, budgets, caps, makers, &rdp_methods),
             rdp_methods);
    }
    if (!needs_candidates)
        return collect(outputs, arms);

    double value = 0;
    const auto budgets = allocate(curves.best, total_budget, &value);
    report("allocation", value, budgets);
    const auto choice = curve_methods(curves.best, budgets);
    const auto candidates =
        make_candidates(objective, budgets, caps, makers, needs_loop ? nullptr : &choice);
    std::optional<Objective::Terms> choice_metrics;
    if (want_choice)
        choice_metrics = keep("BUDGET_CHOICE", candidates, choice);
    if (!needs_loop)
        return collect(outputs, arms);

    Search search(db, objective, candidates, choice);
    const auto checked = checked_loop(objective, candidates, choice, choice_metrics,
                                      search.run(m_seed), "BUDGET_LOOP");
    timed("BUDGET_LOOP");
    if (want_loop)
        keep("BUDGET_LOOP", candidates, checked.methods, checked.metrics);
    if (want_refine) {
        auto trackers = search.trackers();
        auto refined =
            refine_points(db, objective, caps, makers, assemble(candidates, checked.methods),
                          checked.value, checked.reverted ? nullptr : &trackers);
        outputs["BUDGET_LOOP_REFINE"] = {"BUDGET_LOOP_REFINE",
                                         core::to_simplification(refined.kept), refined.value,
                                         refined.metrics};
        timed("BUDGET_LOOP_REFINE");
    }
    return collect(outputs, arms);
}

std::vector<Arm> Teacher::single_recipes(std::size_t total_budget) const {
    const auto budgets = allocation::create_trajectory_budgets(
        m_db, allocation::AllocatorType::SizeProportional, total_budget, 0.0);
    Makers makers(m_db.size());
    const auto candidates = make_candidates(m_objective, budgets, budgets, makers);
    std::vector<Arm> arms;
    for (std::size_t c = 0; c < kCandidates.size(); ++c) {
        const auto name = std::format("RECIPE_{}", kCandidates[c]);
        const auto metrics = m_objective.metrics(candidates[c]);
        arms.push_back(
            {name, core::to_simplification(candidates[c]), m_objective.total(metrics), metrics});
        std::cout << std::format("teacher {}: J_T {:.4f}\n", name, arms.back().objective);
    }
    return arms;
}

} // namespace teacher
