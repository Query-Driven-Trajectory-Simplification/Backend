#include "backend/engine/teacher/teacher.hpp"

#include "backend/engine/allocation/budget_allocator.hpp"
#include "backend/engine/core/parallel.hpp"
#include "backend/engine/selection/rdp.hpp"
#include "backend/engine/selection/sed.hpp"
#include "backend/engine/selection/selector.hpp"
#include "backend/engine/teacher/curves.hpp"
#include "backend/engine/teacher/greedy.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <iostream>
#include <map>
#include <optional>
#include <random>
#include <set>
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

// The classic selectors, then BKH's greedy on range and where-at with different seeds: none,
// SED's or RDP's best half of the interior points
constexpr std::array<std::string_view, kCandidateCount> kCandidates{
    "RDP", "SED", "UNIFORM", "GREEDY", "GREEDY_SED2", "GREEDY_RDP2"};
constexpr std::size_t kMaxResponsePasses = 10;
constexpr std::size_t kKnnWatchFactor = 3; // each query watches its 3k nearest per state
constexpr std::size_t kMaxAllocationRounds = 3;
constexpr double kMinImprovement = 1e-12;
constexpr std::uint64_t kOrderSeedOffset = 7004;
constexpr std::uint64_t kShuffleSeedOffset = 7006;

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

// candidates[c][i]: trajectory i simplified by kCandidates[c] at budgets[i]
using Candidates = std::array<KeptSets, kCandidates.size()>;
using Methods = std::vector<std::size_t>; // index into kCandidates, per trajectory

std::vector<std::size_t> interior(const std::vector<std::size_t> &kept) {
    return kept.size() <= 2 ? std::vector<std::size_t>{}
                            : std::vector<std::size_t>(kept.begin() + 1, kept.end() - 1);
}

// Every candidate of one trajectory at any budget up to cap. Candidate c at budget b depends
// only on (i, c, b, cap), so the value curves and the final kept sets agree. The greedy fill is
// SED, then RDP, then UNIFORM at cap, the same for every b, so one greedy sequence per seed
// serves the whole curve
class CandidateMaker {
  public:
    CandidateMaker(const Objective &objective, std::size_t i, std::size_t cap,
                   bool reuse_sequences = false)
        : m_original(objective.db().trajectory(i)), m_cap(std::min(cap, m_original.size())),
          m_objective(objective), m_index(i), m_reuse_sequences(reuse_sequences) {}

    std::vector<std::size_t> make(std::size_t candidate, std::size_t budget) {
        budget = std::min(budget, m_original.size());
        if (budget > m_cap)
            throw std::runtime_error("teacher: candidate budget above its cap");
        // Every method returns the same kept set at the endpoints and at full resolution.
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
        if (!m_reuse_sequences || type == AlgorithmType::UNIFORM)
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
    // A greedy sequence seeded with interior points selected by the seed algorithm.
    const std::vector<std::size_t> &greedy_sequence(std::size_t seed_size,
                                                    AlgorithmType seed_algorithm) {
        // Classic-only requests do not need the greedy workload index or its fill candidates.
        if (!m_greedy) {
            m_greedy.emplace(m_objective, m_index);
            // SED's and RDP's points in their own order of importance, then UNIFORM's; greedy
            // skips points it already keeps
            m_fill = selection::sed::sequence(m_original, m_cap);
            const auto rdp_order = selection::rdp::sequence(m_original, m_cap);
            m_fill.insert(m_fill.end(), rdp_order.begin(), rdp_order.end());
            const auto uniform =
                selection::simplify_trajectory(m_original, AlgorithmType::UNIFORM, m_cap);
            m_fill.insert(m_fill.end(), uniform.begin(), uniform.end());
        }
        // The sequence depends on its seed, not which candidate requested it. In particular,
        // all unseeded variants and equal-sized SED seeds share the same greedy work.
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
    bool m_reuse_sequences;
    std::optional<Greedy> m_greedy;
    std::optional<std::vector<std::size_t>> m_rdp, m_sed;
    std::vector<std::size_t> m_fill;
    std::map<std::pair<AlgorithmType, std::size_t>, std::vector<std::size_t>> m_greedy_sequences;
};

// Each trajectory's candidates at its budget, or only its selected method when supplied
Candidates make_candidates(const Objective &objective, const std::vector<std::size_t> &budgets,
                           const std::vector<std::size_t> &caps,
                           const Methods *selected = nullptr) {
    Candidates candidates;
    for (auto &kept : candidates)
        kept.resize(budgets.size());
    core::parallel_for(budgets.size(), [&](std::size_t i) {
        // Greedy candidates also need classic selectors for their seeds and fill. Build each
        // classic sequence once when producing all candidates or a greedy candidate.
        CandidateMaker maker(objective, i, caps[i], !selected || (*selected)[i] >= kGreedy);
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

// The best of one trajectory's candidates by independent_value, ties to the lower index. RDP,
// index 0 and always among them, is scored exactly; the rest go best bound first, so the bound
// in independent_value skips most of the losers' kNN checks
struct Pick {
    std::size_t method;
    double value, rdp;
};

Pick pick(const Objective &objective, std::size_t i,
          std::span<const std::pair<std::size_t, core::TrajectoryView>> options,
          std::span<const core::TrajectoryView> partners) {
    const double rdp = objective.independent_value(i, options[0].second, partners);
    Pick best{options[0].first, rdp, rdp};
    std::vector<Objective::IndependentScore> scores;
    scores.reserve(options.size() - 1);
    std::vector<std::pair<double, std::size_t>> order; // bound, position in options
    order.reserve(options.size() - 1);
    for (std::size_t k = 1; k < options.size(); ++k) {
        scores.push_back(objective.prepare_independent(i, options[k].second, partners));
        order.push_back({scores.back().bound, k});
    }
    std::ranges::sort(order, std::greater{});
    for (const auto &[bound, k] : order) {
        if (bound < best.value - 1e-15)
            break;
        const auto c = options[k].first;
        const double v = objective.independent_value(scores[k - 1], best.value);
        if (v > best.value || (v == best.value && c < best.method)) {
            best.value = v;
            best.method = c;
        }
    }
    return best;
}

Methods independent_choice(const core::TrajectoryDb &db, const Objective &objective,
                           const Candidates &candidates, const KeptSets *partners) {
    Methods method(db.size());
    std::optional<PreparedTrajectories> prepared;
    std::vector<core::TrajectoryView> partner_views;
    if (partners) {
        prepared.emplace(db, *partners);
        partner_views = prepared->views();
    }
    core::parallel_for(db.size(), [&](std::size_t i) {
        std::vector<PreparedTrajectory> owned;
        std::vector<std::pair<std::size_t, core::TrajectoryView>> options;
        owned.reserve(kCandidates.size());
        for (std::size_t c = 0; c < kCandidates.size(); ++c) {
            owned.emplace_back(db.trajectory(i), candidates[c][i]);
            options.push_back({c, owned.back().view()});
        }
        method[i] = pick(objective, i, options, partner_views).method;
    });
    return method;
}

// Value curves from 2 up to caps[i] points: the best candidate's value and RDP's alone. Scored
// independently, partners fixed at the given reference state
struct Curves {
    std::vector<Curve> best, rdp;
};

Curves build_curves(const core::TrajectoryDb &db, const Objective &objective,
                    const KeptSets &partners, const std::vector<std::size_t> &caps, bool rdp_only) {
    Curves curves{std::vector<Curve>(db.size()), std::vector<Curve>(db.size())};
    const PreparedTrajectories prepared(db, partners);
    const auto partner_views = prepared.views();
    core::parallel_for(db.size(), [&](std::size_t i) {
        CandidateMaker maker(objective, i, caps[i], true);
        const std::size_t max_b = std::min(caps[i], db.trajectory(i).size());
        auto &best = curves.best[i];
        auto &rdp = curves.rdp[i];
        best.value.reserve(max_b - 1);
        best.method.reserve(max_b - 1);
        rdp.value.reserve(max_b - 1);
        rdp.method.reserve(max_b - 1);
        std::vector<std::vector<std::size_t>> seen;
        std::vector<PreparedTrajectory> owned;
        std::vector<std::pair<std::size_t, core::TrajectoryView>> options;
        seen.reserve(kCandidates.size());
        owned.reserve(kCandidates.size());
        options.reserve(kCandidates.size());
        for (std::size_t b = 2; b <= max_b; ++b) {
            seen.clear();
            options.clear();
            for (std::size_t c = 0; c < (rdp_only ? 1 : kCandidates.size()); ++c) {
                auto kept = maker.make(c, b);
                // A copy of an earlier candidate scores the same, and ties go to the earlier one
                if (std::ranges::find(seen, kept) != seen.end())
                    continue;
                const auto slot = options.size();
                if (slot == owned.size())
                    owned.emplace_back(db.trajectory(i), kept);
                else
                    owned[slot].assign(db.trajectory(i), kept);
                options.push_back({c, owned[slot].view()});
                seen.push_back(std::move(kept));
            }
            const auto chosen = pick(objective, i, options, partner_views);
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

// The loop: coordinate ascent on J_T. One trajectory at a time, in a random order, switches to
// the candidate that raises J_T most given everyone's current choice. Only trajectories with join
// links or, with the ranking kNN term, a watching query can gain, since nothing else depends on
// the state. Every switch strictly raises the tracked J_T, so it stops
// update = false: one pass against the starting state, nobody seeing anyone else's switch
// Built once per start and candidate set; run() copies the trackers, so the loop and its controls
// can share the setup
class Search {
  public:
    Search(const core::TrajectoryDb &db, const Objective &objective, const Candidates &candidates,
           const Methods &method)
        : m_start(method) {
        // With the ranking kNN term, a trajectory also matters to the queries that watch it
        if (objective.config().ranking_knn)
            m_knn.emplace(objective, assemble(candidates, method), candidates,
                          kKnnWatchFactor * objective.workloads().k);
        const auto active = [&](std::size_t i) {
            return !objective.links(i).empty() || (m_knn && m_knn->watched(i));
        };
        for (std::size_t i = 0; i < db.size(); ++i)
            if (active(i))
                m_order.push_back(i);
        m_prepared.resize(db.size());
        for (auto &v : m_views)
            v.resize(db.size());
        m_separable.resize(db.size());
        m_representative.resize(db.size());
        m_knn_distances.resize(db.size());
        core::parallel_for(db.size(), [&](std::size_t i) {
            auto &owned = m_prepared[i];
            const auto original = db.trajectory(i);
            if (!active(i)) {
                // Isolated trajectories never switch; only their selected state is needed.
                owned.emplace_back(original, candidates[method[i]][i]);
                m_views[method[i]][i] = owned.back().view();
                return;
            }
            // Reserve before taking borrowed views, so adding another candidate cannot move them.
            owned.reserve(kCandidates.size());
            for (std::size_t c = 0; c < kCandidates.size(); ++c) {
                std::size_t first = 0;
                while (first < c && candidates[first][i] != candidates[c][i])
                    ++first;
                m_representative[i][c] = first;
                if (first == c) {
                    owned.emplace_back(original, candidates[c][i]);
                    m_views[c][i] = owned.back().view();
                    m_separable[i][c] = objective.separable(i, m_views[c][i]);
                    if (m_knn)
                        m_knn_distances[i][c] = m_knn->distances(i, m_views[c][i]);
                } else {
                    m_views[c][i] = m_views[first][i];
                    m_separable[i][c] = m_separable[i][first];
                }
            }
        });
        m_state.resize(db.size());
        for (std::size_t i = 0; i < db.size(); ++i)
            m_state[i] = m_views[method[i]][i];
        m_join.emplace(objective, m_state);
    }
    Search(const Search &) = delete;
    Search &operator=(const Search &) = delete;

    // update = false: one pass against the starting state, nobody seeing anyone else's switch
    Methods run(std::uint64_t seed, bool update = true) const {
        auto method = m_start;
        auto state = m_state;
        auto join = *m_join;
        auto knn = m_knn;
        auto order = m_order;
        std::mt19937_64 rng(seed + kOrderSeedOffset);

        for (std::size_t pass = 1; pass <= kMaxResponsePasses; ++pass) {
            std::ranges::shuffle(order, rng);
            std::size_t switches = 0;
            for (const auto i : order) {
                const double here = m_separable[i][method[i]];
                std::size_t best = method[i];
                JoinState::Proposal best_proposal;
                double best_gain = kMinImprovement; // strict gains only, so ties cannot cycle
                for (std::size_t c = 0; c < kCandidates.size(); ++c) {
                    if (m_representative[i][c] != c ||
                        m_representative[i][c] == m_representative[i][method[i]])
                        continue;
                    auto proposal = join.propose(i, m_views[c][i], state);
                    const double knn_gain = knn ? knn->propose(i, m_knn_distances[i][c]) : 0.0;
                    const double gain = m_separable[i][c] - here + proposal.gain + knn_gain;
                    if (gain > best_gain) {
                        best = c;
                        best_gain = gain;
                        best_proposal = std::move(proposal);
                    }
                }
                if (best == method[i])
                    continue;
                method[i] = best;
                ++switches;
                if (!update)
                    continue;
                state[i] = m_views[best][i];
                join.apply(best_proposal);
                if (knn)
                    knn->apply(i, m_knn_distances[i][best]);
            }
            std::cout << std::format("teacher:   pass {}: {} switches, join F1 {:.4f}{}\n", pass,
                                     switches, join.metric(),
                                     knn ? std::format(", knn {:.4f}", knn->metric()) : "");
            if (switches == 0 || !update)
                break;
        }
        return method;
    }

  private:
    Methods m_start;
    std::vector<std::size_t> m_order;
    std::vector<std::vector<PreparedTrajectory>> m_prepared;
    std::array<std::vector<core::TrajectoryView>, kCandidates.size()> m_views;
    // These terms depend only on the candidate, never on the partners' current choices
    std::vector<std::array<double, kCandidates.size()>> m_separable;
    std::vector<std::array<std::size_t, kCandidates.size()>> m_representative;
    std::vector<std::array<std::vector<double>, kCandidates.size()>> m_knn_distances;
    std::optional<KnnState> m_knn;
    std::vector<core::TrajectoryView> m_state;
    std::optional<JoinState> m_join;
};

Methods best_response(const core::TrajectoryDb &db, const Objective &objective,
                      const Candidates &candidates, Methods method, std::uint64_t seed,
                      bool update = true) {
    return Search(db, objective, candidates, method).run(seed, update);
}

// The same workloads with trajectories relabelled at random among those that start and end in the
// same 1 km cells, in every join pair and every kNN query's true neighbours. Each pair keeps its
// kind and seen flag and joins two trajectories that could plausibly match, so the loop on them
// is the same search with wrong but believable partners and neighbours
TeacherWorkloads shuffled_partners(const core::TrajectoryDb &db, const TeacherWorkloads &workloads,
                                   std::uint64_t seed) {
    constexpr double kCell = 1000.0;
    std::map<std::array<long, 4>, std::vector<std::uint32_t>> buckets;
    for (std::size_t i = 0; i < db.size(); ++i) {
        const auto v = db.trajectory(i);
        auto cell = [&](double c) { return static_cast<long>(std::floor(c / kCell)); };
        buckets[{cell(v.x.front()), cell(v.y.front()), cell(v.x.back()), cell(v.y.back())}]
            .push_back(static_cast<std::uint32_t>(i));
    }
    std::vector<std::uint32_t> label(db.size());
    std::mt19937_64 rng(seed + kShuffleSeedOffset);
    for (auto &[key, members] : buckets) {
        auto shuffled = members;
        std::ranges::shuffle(shuffled, rng);
        for (std::size_t k = 0; k < members.size(); ++k)
            label[members[k]] = shuffled[k];
    }

    auto result = workloads;
    std::set<std::pair<std::uint32_t, std::uint32_t>> real;
    for (const auto &p : workloads.positives)
        real.insert({p.i, p.j});
    std::size_t kept_real = 0;
    std::vector<std::vector<JoinPair> *> lists{&result.positives, &result.negatives};
    for (auto &extra : result.extra_joins) {
        lists.push_back(&extra.positives);
        lists.push_back(&extra.negatives);
    }
    for (auto *pairs : lists)
        for (auto &p : *pairs) {
            const auto a = label[p.i], b = label[p.j];
            p.i = std::min(a, b);
            p.j = std::max(a, b);
            kept_real += pairs == &result.positives && real.contains({p.i, p.j});
        }
    std::size_t kept_neighbours = 0, neighbours = 0;
    for (std::size_t e = 0; e < result.knn.size(); ++e) {
        std::vector<std::uint32_t> relabelled;
        for (const auto j : result.knn[e].top) {
            const auto to = label[j] == e ? j : label[j];
            if (std::ranges::find(relabelled, to) == relabelled.end())
                relabelled.push_back(to);
            kept_neighbours += to == j;
            ++neighbours;
        }
        result.knn[e].top = std::move(relabelled);
    }
    std::cout << std::format(
        "teacher: shuffled in {} buckets, {} of {} positive pairs and {} of {} "
        "kNN neighbours still real\n",
        buckets.size(), kept_real, result.positives.size(), kept_neighbours, neighbours);
    return result;
}

core::Simplification to_simplification(const KeptSets &kept) {
    core::Simplification s;
    for (const auto &k : kept)
        s.add_trajectory(k);
    return s;
}

std::string wins(const Methods &method) {
    std::array<std::size_t, kCandidates.size()> count{};
    for (const auto m : method)
        ++count[m];
    std::string out;
    for (std::size_t c = 0; c < kCandidates.size(); ++c)
        out += std::format("{}{} {}", c ? ", " : "", kCandidates[c], count[c]);
    return out;
}

// The requested arms, in the order asked
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

Teacher::Teacher(const core::TrajectoryDb &db, TeacherWorkloads workloads,
                 const ObjectiveConfig &config, std::uint64_t seed)
    : m_db(db), m_workloads(std::move(workloads)), m_objective(db, m_workloads, config),
      m_seed(seed) {
    const auto &f = m_objective.floors();
    std::cout << std::format("teacher: {} join pairs ({} positive), floors knn {:.4f}, range "
                             "{:.4f}, where-at {:.4f}, join {:.4f}\n",
                             m_objective.pair_count(), m_objective.positive_count(),
                             f[Objective::Knn], f[Objective::Range], f[Objective::WhereAt],
                             f[Objective::Join]);
}

std::vector<Arm> Teacher::run(std::size_t total_budget, std::span<const std::string> arms) const {
    for (const auto &a : arms)
        if (!is_arm(a))
            throw std::invalid_argument("teacher: unknown arm " + a);
    const auto requested = [&](std::string_view name) {
        return std::ranges::find(arms, name) != arms.end();
    };
    const bool want_choice = requested("EQUAL_CHOICE");
    const bool want_guess = requested("EQUAL_CHOICE_RDP_PARTNERS");
    const bool want_equal_loop = requested("EQUAL_LOOP");
    const bool want_budget_rdp = requested("BUDGET_RDP");
    const bool want_budget_choice = requested("BUDGET_CHOICE");
    const bool want_loop = requested("BUDGET_LOOP");
    const bool want_reallocate = requested("BUDGET_LOOP_REALLOCATE");
    const bool want_shuffled = requested("BUDGET_LOOP_SHUFFLED");
    const bool want_no_update = requested("BUDGET_LOOP_NO_UPDATE");
    const bool needs_loop_candidates =
        want_loop || want_reallocate || want_shuffled || want_no_update;
    const bool needs_budget_candidates = want_budget_choice || needs_loop_candidates;
    const bool needs_curves = want_budget_rdp || needs_budget_candidates;

    const auto &db = m_db;
    const auto &objective = m_objective;
    const auto equal_budgets = allocation::create_trajectory_budgets(
        db, allocation::AllocatorType::SizeProportional, total_budget, 0.0);
    // Curves up to about four times the mean interior budget, at least 6 interior points, and
    // never below a trajectory's equal share
    const double mean_interior =
        static_cast<double>(total_budget - 2 * db.size()) / static_cast<double>(db.size());
    const auto cap = 2 + static_cast<std::size_t>(std::ceil(std::max(6.0, 4.0 * mean_interior)));
    std::vector<std::size_t> caps(db.size());
    for (std::size_t i = 0; i < db.size(); ++i)
        caps[i] = std::min(std::max(cap, equal_budgets[i]), db.trajectory(i).size());

    std::map<std::string, Arm, std::less<>> outputs;
    auto keep = [&](std::string_view name, const Candidates &from, const Methods &method) {
        auto kept = assemble(from, method);
        const double j_t = objective.total(kept);
        std::cout << std::format("teacher {}: {}, J_T {:.4f}\n", name, wins(method), j_t);
        outputs[std::string(name)] = {std::string(name), to_simplification(kept), j_t};
    };
    auto start = std::chrono::steady_clock::now();
    auto timed = [&](std::string_view step) {
        std::cout << std::format("teacher: {} {:.1f} s\n", step, seconds_since(start));
        start = std::chrono::steady_clock::now();
    };
    const Methods rdp_methods(db.size(), kRdp);
    const bool equal_alternatives = want_choice || want_guess || want_equal_loop;
    const auto equal_candidates = make_candidates(objective, equal_budgets, caps,
                                                  equal_alternatives ? nullptr : &rdp_methods);
    timed("candidates");

    if (want_choice) {
        keep("EQUAL_CHOICE", equal_candidates,
             independent_choice(db, objective, equal_candidates, nullptr));
        timed("EQUAL_CHOICE");
    }
    if (!(want_guess || want_equal_loop || needs_curves))
        return collect(outputs, arms);

    // Partners are fixed at RDP with equal budgets, so trajectories score independently
    Curves curves;
    if (needs_curves) {
        const bool rdp_only = !(want_guess || want_equal_loop || needs_budget_candidates);
        curves = build_curves(db, objective, equal_candidates[kRdp], caps, rdp_only);
        timed(std::format("value curves up to {} points", cap));
    }
    if (want_guess || want_equal_loop) {
        // Fixed budgets only need scores at those budgets, not complete value curves
        const auto guess = needs_curves ? curve_methods(curves.best, equal_budgets)
                                        : independent_choice(db, objective, equal_candidates,
                                                             &equal_candidates[kRdp]);
        if (want_guess)
            keep("EQUAL_CHOICE_RDP_PARTNERS", equal_candidates, guess);
        if (want_equal_loop) {
            keep("EQUAL_LOOP", equal_candidates,
                 best_response(db, objective, equal_candidates, guess, m_seed));
            timed("EQUAL_LOOP");
        }
    }
    auto report = [&](std::string_view what, double value,
                      const std::vector<std::size_t> &budgets) {
        std::size_t at_cap = 0;
        for (std::size_t i = 0; i < budgets.size(); ++i)
            at_cap += budgets[i] == caps[i] && caps[i] < db.trajectory(i).size();
        std::cout << std::format("teacher: {}, curve sum {:.4f}, {} trajectories at the cap\n",
                                 what, value, at_cap);
    };
    if (want_budget_rdp) {
        double value = 0;
        const auto budgets = allocate(curves.rdp, total_budget, &value);
        report("RDP allocation", value, budgets);
        keep("BUDGET_RDP", make_candidates(objective, budgets, caps, &rdp_methods), rdp_methods);
    }
    if (needs_budget_candidates) {
        double value = 0;
        const auto budgets = allocate(curves.best, total_budget, &value);
        report("allocation", value, budgets);
        const auto choice = curve_methods(curves.best, budgets);
        const auto candidates =
            make_candidates(objective, budgets, caps, needs_loop_candidates ? nullptr : &choice);
        if (want_budget_choice)
            keep("BUDGET_CHOICE", candidates, choice);
        // The loop and its no-update control start from the same state and share the setup
        std::optional<Search> search;
        if (want_loop || want_reallocate || want_no_update)
            search.emplace(db, objective, candidates, choice);
        if (want_loop || want_reallocate) {
            const auto looped = search->run(m_seed);
            timed("BUDGET_LOOP");
            if (want_loop)
                keep("BUDGET_LOOP", candidates, looped);
            if (want_reallocate) {
                // Each round's curves see the partners as the last round left them. A round is
                // kept only if J_T rises
                auto current = assemble(candidates, looped);
                auto current_methods = looped;
                double current_value = objective.total(current);
                for (std::size_t round = 1; round <= kMaxAllocationRounds; ++round) {
                    const auto round_curves = build_curves(db, objective, current, caps, false);
                    double round_value = 0;
                    const auto round_budgets =
                        allocate(round_curves.best, total_budget, &round_value);
                    const auto next_candidates = make_candidates(objective, round_budgets, caps);
                    auto next_methods = best_response(
                        db, objective, next_candidates,
                        curve_methods(round_curves.best, round_budgets), m_seed + round);
                    auto next = assemble(next_candidates, next_methods);
                    const double next_value = objective.total(next);
                    std::cout << std::format(
                        "teacher: reallocation round {}, J_T {:.4f} -> {:.4f}\n", round,
                        current_value, next_value);
                    if (next_value <= current_value + kMinImprovement)
                        break;
                    current_value = next_value;
                    current = std::move(next);
                    current_methods = std::move(next_methods);
                }
                std::cout << std::format("teacher BUDGET_LOOP_REALLOCATE: {}, J_T {:.4f}\n",
                                         wins(current_methods), current_value);
                outputs["BUDGET_LOOP_REALLOCATE"] = {"BUDGET_LOOP_REALLOCATE",
                                                     to_simplification(current), current_value};
                timed("BUDGET_LOOP_REALLOCATE");
            }
        }
        if (want_no_update) {
            keep("BUDGET_LOOP_NO_UPDATE", candidates, search->run(m_seed, false));
            timed("BUDGET_LOOP_NO_UPDATE");
        }
        if (want_shuffled) {
            // Searches on fake partners and neighbours; keep scores the result on the real ones
            const auto shuffled = shuffled_partners(db, m_workloads, m_seed);
            const Objective fake(db, shuffled, objective.config());
            keep("BUDGET_LOOP_SHUFFLED", candidates,
                 best_response(db, fake, candidates, choice, m_seed));
            timed("BUDGET_LOOP_SHUFFLED");
        }
    }
    return collect(outputs, arms);
}

std::vector<Arm> Teacher::single_recipes(std::size_t total_budget) const {
    const auto budgets = allocation::create_trajectory_budgets(
        m_db, allocation::AllocatorType::SizeProportional, total_budget, 0.0);
    const auto candidates = make_candidates(m_objective, budgets, budgets);
    std::vector<Arm> arms;
    for (std::size_t c = 0; c < kCandidates.size(); ++c) {
        const auto name = std::format("RECIPE_{}", kCandidates[c]);
        arms.push_back({name, to_simplification(candidates[c]), m_objective.total(candidates[c])});
        std::cout << std::format("teacher {}: J_T {:.4f}\n", name, arms.back().objective);
    }
    return arms;
}

} // namespace teacher
