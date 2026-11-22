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
    kGreedySedHalf,
    kGreedyRdpHalf,
    kCandidateCount,
};

// Classic selectors, then greedy seeded with nothing, or SED's or RDP's best half
constexpr std::array<std::string_view, kCandidateCount> kCandidates{
    "RDP", "SED", "UNIFORM", "GREEDY", "GREEDY_SED2", "GREEDY_RDP2"};
constexpr std::size_t kMaxResponsePasses = 10;
constexpr std::size_t kKnnWatchFactor = 3;
constexpr std::array<std::size_t, 3> kTradeSteps{1, 2, 4}; // points moved per trade
constexpr std::size_t kMaxTradeRounds = 3;
constexpr double kMinImprovement = 1e-12;

// candidates[c][i]: trajectory i simplified by kCandidates[c] at budgets[i]
using Candidates = std::array<KeptSets, kCandidates.size()>;
using Methods = std::vector<std::size_t>; // per trajectory

std::vector<std::size_t> interior(const std::vector<std::size_t> &kept) {
    return kept.size() <= 2 ? std::vector<std::size_t>{}
                            : std::vector<std::size_t>(kept.begin() + 1, kept.end() - 1);
}

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
        const std::size_t interior_budget = budget - 2;
        switch (candidate) {
        case kRdp:
            return simplify(AlgorithmType::RDP, budget);
        case kSed:
            return simplify(AlgorithmType::SED, budget);
        case kUniform:
            return simplify(AlgorithmType::UNIFORM, budget);
        case kGreedy:
            return prefix(greedy_sequence(0, AlgorithmType::SED), budget);
        case kGreedySedHalf:
            return prefix(greedy_sequence(interior_budget / 2, AlgorithmType::SED), budget);
        case kGreedyRdpHalf:
            return prefix(greedy_sequence(interior_budget / 2, AlgorithmType::RDP), budget);
        default:
            throw std::logic_error("teacher: unknown candidate");
        }
    }

  private:
    std::vector<std::size_t> simplify(AlgorithmType type, std::size_t b) {
        if (type == AlgorithmType::UNIFORM)
            return selection::simplify_trajectory(m_original, type, b);
        auto &cached = type == AlgorithmType::RDP ? m_rdp : m_sed;
        if (!cached)
            cached = type == AlgorithmType::RDP ? selection::rdp::sequence(m_original, m_cap)
                                                : selection::sed::sequence(m_original, m_cap);
        return prefix(*cached, b);
    }
    static std::vector<std::size_t> prefix(const std::vector<std::size_t> &sequence,
                                           std::size_t b) {
        std::vector<std::size_t> kept(sequence.begin(),
                                      sequence.begin() + static_cast<std::ptrdiff_t>(b));
        std::ranges::sort(kept);
        return kept;
    }
    const std::vector<std::size_t> &greedy_sequence(std::size_t seed_size,
                                                    AlgorithmType seed_algorithm) {
        if (!m_greedy) {
            m_greedy.emplace(m_objective, m_index);
            // SED's order of importance covers every point up to the cap
            m_fill = selection::sed::sequence(m_original, m_cap);
        }
        const auto seed_type = seed_size ? seed_algorithm : AlgorithmType::SED;
        auto &cached = m_greedy_sequences[{seed_type, seed_size}];
        if (cached.empty()) {
            const auto seed = seed_size ? interior(simplify(seed_algorithm, 2 + seed_size))
                                        : std::vector<std::size_t>{};
            cached = m_greedy->sequence(m_cap, seed, m_fill);
        }
        return cached;
    }

    core::TrajectoryView m_original;
    std::size_t m_cap;
    const Objective &m_objective;
    std::size_t m_index;
    std::optional<Greedy> m_greedy;
    std::optional<std::vector<std::size_t>> m_rdp, m_sed;
    std::vector<std::size_t> m_fill;
    std::map<std::pair<AlgorithmType, std::size_t>, std::vector<std::size_t>> m_greedy_sequences;
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
          std::span<const core::TrajectoryView> partners, std::span<const double> range_weights) {
    const double rdp = objective.independent_value(
        i, options[0].second, partners, -std::numeric_limits<double>::infinity(), range_weights);
    Pick best{options[0].first, rdp, rdp};
    std::vector<Objective::IndependentScore> scores;
    scores.reserve(options.size() - 1);
    std::vector<std::pair<double, std::size_t>> order; // bound, position in options
    order.reserve(options.size() - 1);
    for (std::size_t k = 1; k < options.size(); ++k) {
        scores.push_back(
            objective.prepare_independent(i, options[k].second, partners, range_weights));
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
            const auto chosen = pick(objective, i, options, partner_views, range_weights);
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
                   m_knn->watched(i) || objective.scale(Objective::Shape) > 0 ||
                   objective.scale(Objective::Margin) > 0;
        };
        for (std::size_t i = 0; i < db.size(); ++i)
            if (active(i))
                m_order.push_back(i);
        std::cout << std::format("teacher:   loop setup: prepare {:.1f} s\n", prepare_s);
        m_join.emplace(objective, m_state);
        m_range.emplace(objective, m_state);
        // Every candidate's separable value, kNN distances, box coverage and join outcomes; none
        // depend on the others
        m_separable.resize(db.size());
        m_knn_distances.resize(db.size());
        m_coverage.resize(db.size());
        m_outcomes.resize(db.size());
        core::parallel_for(db.size(), [&](std::size_t i) {
            if (!active(i))
                return;
            for (std::size_t c = 0; c < kCandidates.size(); ++c) {
                const auto first = m_representative[i][c];
                m_separable[i][c] =
                    first == c ? objective.separable(i, m_views[c][i]) : m_separable[i][first];
                if (first != c)
                    continue;
                // The start's distances are already in the tracker
                m_knn_distances[i][c] = c == m_representative[i][method[i]]
                                            ? m_knn->stored(i)
                                            : m_knn->distances(i, m_views[c][i]);
                m_coverage[i][c] = m_range->coverage(i, m_views[c][i]);
                m_outcomes[i][c] = m_join->outcomes(i, m_views[c][i]);
            }
        });
        std::cout << std::format("teacher:   loop setup: kNN lists {:.1f} s, candidates {:.1f} s\n",
                                 knn_s, core::seconds_since(start));
    }
    Search(const Search &) = delete;
    Search &operator=(const Search &) = delete;

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
                    const double gain = m_separable[i][c] - here + proposal.gain +
                                        range_proposal.gain + knn.propose(i, m_knn_distances[i][c]);
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
                knn.apply(i, m_knn_distances[i][best]);
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
    std::vector<std::array<double, kCandidates.size()>> m_separable;
    std::vector<std::array<std::size_t, kCandidates.size()>> m_representative;
    std::vector<std::array<std::vector<double>, kCandidates.size()>> m_knn_distances;
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
};
Checked checked_loop(const Objective &objective, const Candidates &candidates, const Methods &start,
                     std::optional<double> before, Methods looped, std::string_view name) {
    if (!before)
        before = objective.total(assemble(candidates, start));
    const double after = objective.total(assemble(candidates, looped));
    const bool lost = after < *before;
    std::cout << std::format("teacher: {} exact check, J_T {:.6f} -> {:.6f}{}\n", name, *before,
                             after, lost ? ", lost: keeping the start" : "");
    if (lost)
        return {start, *before};
    return {std::move(looped), after};
}

struct Traded {
    KeptSets kept;
    double value;
    std::optional<Objective::Terms> metrics;
};

// Moves delta points between trajectories while exact J_T rises. Each round scores every
// trajectory's best recipe at b + delta and b - delta against the current state, pairs the best
// buyers with the cheapest sellers, and keeps a pair only if its exact gain, rescored when applied,
// is positive. Each round the kNN lists are extended to watch the offered versions
Traded trade_budgets(const core::TrajectoryDb &db, const Objective &objective,
                     const std::vector<std::size_t> &caps, Makers &makers, KeptSets kept,
                     double start_value) {
    const std::size_t n = db.size();
    std::vector<PreparedTrajectory> owned;
    owned.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        owned.emplace_back(db.trajectory(i), kept[i]);
    std::vector<core::TrajectoryView> state(n);
    for (std::size_t i = 0; i < n; ++i)
        state[i] = owned[i].view();
    JoinState join(objective, state);
    RangeState range(objective, state);
    std::vector<double> separable(n), state_error(n);
    core::parallel_for(n, [&](std::size_t i) {
        separable[i] = objective.separable(i, state[i]);
        state_error[i] = objective.deviation(i, state[i]);
    });
    // Built once: the trades keep it current
    KnnState knn(objective, kept, kKnnWatchFactor * objective.workloads().k);

    struct Pending {
        std::size_t i;
        PreparedTrajectory prepared;
        std::vector<std::size_t> kept;
        double separable = 0.0, gain = 0.0;
        JoinState::Proposal join;
        RangeState::Proposal range;
        std::vector<double> distances;
    };
    auto pending = [&](std::size_t i, std::vector<std::size_t> next) {
        Pending p{i, PreparedTrajectory(db.trajectory(i), next), std::move(next), 0.0, 0.0, {}, {},
                  {}};
        const auto view = p.prepared.view();
        p.separable = objective.separable(i, view);
        p.join = join.propose(i, view);
        p.range = range.propose(i, view);
        p.gain = p.separable - separable[i] + p.join.gain + p.range.gain;
        // One trade at a time, so the distances themselves go in parallel
        p.distances = knn.distances(i, view, true);
        p.gain += knn.propose(i, p.distances);
        return p;
    };
    auto commit = [&](Pending &p) {
        const auto i = p.i;
        join.apply(p.join);
        range.apply(p.range);
        knn.apply(i, p.distances);
        owned[i] = std::move(p.prepared);
        state[i] = owned[i].view();
        kept[i] = std::move(p.kept);
        separable[i] = p.separable;
        state_error[i] = objective.deviation(i, state[i]);
    };

    struct Offer {
        double gain = -std::numeric_limits<double>::infinity();
        std::vector<std::size_t> kept;
    };
    const auto start_kept = kept;
    double value = start_value;
    std::cout << std::format("teacher: trading from J_T {:.6f}\n", value);
    // Sequences depend only on the trajectory and its cap, so are built once for every round
    for (std::size_t round = 1; round <= kMaxTradeRounds; ++round) {
        auto start = std::chrono::steady_clock::now();
        // buy[d][i], sell[d][i]: i's best exact change at b_i + delta and b_i - delta
        std::array<std::vector<Offer>, kTradeSteps.size()> buy, sell;
        for (std::size_t d = 0; d < kTradeSteps.size(); ++d) {
            buy[d].resize(n);
            sell[d].resize(n);
        }
        core::parallel_for(n, [&](std::size_t i) {
            const std::size_t b = kept[i].size();
            const std::size_t top = std::min(caps[i], db.trajectory(i).size());
            auto &maker = maker_for(makers, objective, i, caps[i]);
            auto best_at = [&](std::size_t budget) {
                Offer best;
                std::vector<std::vector<std::size_t>> seen;
                for (std::size_t c = 0; c < kCandidates.size(); ++c) {
                    auto candidate = maker.make(c, budget);
                    if (std::ranges::find(seen, candidate) != seen.end())
                        continue;
                    seen.push_back(candidate);
                    const PreparedTrajectory prepared(db.trajectory(i), candidate);
                    const auto view = prepared.view();
                    double gain = objective.separable(i, view) - separable[i] +
                                  join.propose(i, view).gain + range.propose(i, view).gain;
                    gain +=
                        knn.propose_bounded(i, view, state_error[i] + objective.deviation(i, view));
                    if (gain > best.gain)
                        best = {gain, std::move(candidate)};
                }
                return best;
            };
            for (std::size_t d = 0; d < kTradeSteps.size(); ++d) {
                if (b + kTradeSteps[d] <= top)
                    buy[d][i] = best_at(b + kTradeSteps[d]);
                if (b >= 2 + kTradeSteps[d])
                    sell[d][i] = best_at(b - kTradeSteps[d]);
            }
        });
        const double sweep_s = core::seconds_since(start);
        start = std::chrono::steady_clock::now();
        // The trades below try only these states, so the kNN lists watch them. Either way the
        // lists can now see trajectories the old ones missed, which changes the tracked metric
        // of the current state; that change is carried into the tracked total below
        const double before_watch = knn.metric();
        {
            std::vector<std::vector<PreparedTrajectory>> prepared(n);
            std::vector<std::vector<core::TrajectoryView>> versions(n);
            core::parallel_for(n, [&](std::size_t i) {
                for (const auto *offers : {&buy, &sell})
                    for (const auto &by_trajectory : *offers)
                        if (std::isfinite(by_trajectory[i].gain))
                            prepared[i].emplace_back(db.trajectory(i), by_trajectory[i].kept);
                for (const auto &version : prepared[i])
                    versions[i].push_back(version.view());
            });
            knn.watch(versions, state);
        }
        const double knn_correction =
            objective.scale(Objective::Knn) * (knn.metric() - before_watch);
        value += knn_correction;
        const double watch_s = core::seconds_since(start);
        start = std::chrono::steady_clock::now();

        std::vector<char> used(n, 0);
        std::size_t profitable = 0, accepted = 0;
        const double before = value;
        for (std::size_t d = 0; d < kTradeSteps.size(); ++d) {
            std::vector<std::pair<double, std::size_t>> buyers, sellers;
            for (std::size_t i = 0; i < n; ++i) {
                if (std::isfinite(buy[d][i].gain))
                    buyers.push_back({buy[d][i].gain, i});
                if (std::isfinite(sell[d][i].gain))
                    sellers.push_back({sell[d][i].gain, i});
            }
            std::ranges::sort(buyers, std::greater{});
            std::ranges::sort(sellers, std::greater{});
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
                ++profitable;
                // Exact: the seller first, then the buyer against the changed state
                auto old_kept = kept[seller];
                auto first = pending(seller, std::move(sell[d][seller].kept));
                commit(first);
                auto second = pending(r, std::move(buy[d][r].kept));
                if (first.gain + second.gain > kMinImprovement) {
                    commit(second);
                    value += first.gain + second.gain;
                    ++accepted;
                } else {
                    auto back = pending(seller, std::move(old_kept));
                    commit(back);
                }
            }
        }
        std::cout << std::format(
            "teacher: trade round {}: {} profitable pairs, {} kept, J_T {:.6f} "
            "-> {:.6f}, kNN list correction {:+.6f}, sweep {:.1f} s, watch {:.1f} s, trades "
            "{:.1f} s\n",
            round, profitable, accepted, before, value, knn_correction, sweep_s, watch_s,
            core::seconds_since(start));
        if (accepted == 0)
            break;
    }
    const auto terms = objective.metrics(kept);
    const double exact = objective.total(terms);
    // Where-at is rescored on every commit, so only these three can drift
    std::cout << std::format(
        "teacher: trading J_T tracked {:.6f}, exact {:.6f}; tracked/exact kNN {:.6f}/{:.6f}, "
        "range {:.6f}/{:.6f}, join {:.6f}/{:.6f}\n",
        value, exact, knn.metric(), terms[Objective::Knn], range.metric(), terms[Objective::Range],
        join.metric(), terms[Objective::Join]);
    // The kNN tracker is not exact everywhere, so the trades could in principle lose
    if (exact < start_value)
        return {start_kept, start_value, {}};
    return {std::move(kept), exact, terms};
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
    const bool want_trade = requested("BUDGET_LOOP_TRADE");
    const bool needs_loop = want_loop || want_trade;
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
    // Returns the arm's exact J_T; known skips recomputing it
    auto keep = [&](std::string_view name, const Candidates &from, const Methods &method,
                    std::optional<double> known = {}) {
        auto kept = assemble(from, method);
        std::optional<Objective::Terms> metrics;
        if (!known)
            metrics = objective.metrics(kept);
        const double j_t = known ? *known : objective.total(*metrics);
        std::cout << std::format("teacher {}: {}, J_T {:.4f}\n", name, wins(method), j_t);
        outputs[std::string(name)] = {std::string(name), core::to_simplification(kept), j_t,
                                      metrics};
        return j_t;
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
    std::optional<double> choice_value;
    if (want_choice)
        choice_value = keep("BUDGET_CHOICE", candidates, choice);
    if (!needs_loop)
        return collect(outputs, arms);

    const auto checked =
        checked_loop(objective, candidates, choice, choice_value,
                     Search(db, objective, candidates, choice).run(m_seed), "BUDGET_LOOP");
    timed("BUDGET_LOOP");
    if (want_loop)
        keep("BUDGET_LOOP", candidates, checked.methods, checked.value);
    if (want_trade) {
        auto traded = trade_budgets(db, objective, caps, makers,
                                    assemble(candidates, checked.methods), checked.value);
        outputs["BUDGET_LOOP_TRADE"] = {"BUDGET_LOOP_TRADE", core::to_simplification(traded.kept),
                                        traded.value, traded.metrics};
        timed("BUDGET_LOOP_TRADE");
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
        arms.push_back(
            {name, core::to_simplification(candidates[c]), m_objective.total(candidates[c]), {}});
        std::cout << std::format("teacher {}: J_T {:.4f}\n", name, arms.back().objective);
    }
    return arms;
}

} // namespace teacher
