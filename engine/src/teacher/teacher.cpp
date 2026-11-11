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
    kGreedySedQuarter,
    kGreedySedHalf,
    kGreedyRdpHalf,
    kCandidateCount,
};

// The classic selectors, then BKH's greedy on range and where-at with different seeds: none,
// SED's best quarter or half of the interior points, RDP's best half
constexpr std::array<std::string_view, kCandidateCount> kCandidates{
    "RDP", "SED", "UNIFORM", "GREEDY", "GREEDY_SED4", "GREEDY_SED2", "GREEDY_RDP2"};
constexpr std::size_t kMaxResponsePasses = 10;
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
        case kGreedySedQuarter:
            return prefix(greedy_sequence(interior_budget / 4, AlgorithmType::SED), budget);
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
            for (const auto type :
                 {AlgorithmType::SED, AlgorithmType::RDP, AlgorithmType::UNIFORM}) {
                const auto fill_points = simplify(type, m_cap);
                m_fill.insert(m_fill.end(), fill_points.begin(), fill_points.end());
            }
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

// Coordinate ascent on the exact J_T: one trajectory at a time, in a random order, switches to
// the candidate that raises J_T most given everyone's current choice. Only trajectories with join
// candidates can gain, since nothing else depends on the state. Every switch strictly raises
// J_T, so it stops
Methods best_response(const core::TrajectoryDb &db, const Objective &objective,
                      const Candidates &candidates, Methods method, std::uint64_t seed) {
    std::vector<std::size_t> order;
    for (std::size_t i = 0; i < db.size(); ++i)
        if (!objective.links(i).empty())
            order.push_back(i);
    std::vector<std::vector<PreparedTrajectory>> prepared(db.size());
    std::array<std::vector<core::TrajectoryView>, kCandidates.size()> views;
    for (auto &v : views)
        v.resize(db.size());
    // These terms depend only on the candidate, never on the partners' current choices.
    std::vector<std::array<double, kCandidates.size()>> separable(db.size());
    std::vector<std::array<std::size_t, kCandidates.size()>> representative(db.size());
    core::parallel_for(db.size(), [&](std::size_t i) {
        auto &owned = prepared[i];
        const auto original = db.trajectory(i);
        if (objective.links(i).empty()) {
            // Isolated trajectories never switch; only their selected state is needed.
            owned.emplace_back(original, candidates[method[i]][i]);
            views[method[i]][i] = owned.back().view();
            return;
        }
        // Reserve before taking borrowed views, so adding another candidate cannot move them.
        owned.reserve(kCandidates.size());
        for (std::size_t c = 0; c < kCandidates.size(); ++c) {
            std::size_t first = 0;
            while (first < c && candidates[first][i] != candidates[c][i])
                ++first;
            representative[i][c] = first;
            if (first == c) {
                owned.emplace_back(original, candidates[c][i]);
                views[c][i] = owned.back().view();
                separable[i][c] = objective.separable(i, views[c][i]);
            } else {
                views[c][i] = views[first][i];
                separable[i][c] = separable[i][first];
            }
        }
    });
    std::mt19937_64 rng(seed + kOrderSeedOffset);
    std::vector<core::TrajectoryView> state(db.size());
    for (std::size_t i = 0; i < db.size(); ++i)
        state[i] = views[method[i]][i];
    JoinState join(objective, state);

    for (std::size_t pass = 1; pass <= kMaxResponsePasses; ++pass) {
        std::ranges::shuffle(order, rng);
        std::size_t switches = 0;
        for (const auto i : order) {
            const double here = separable[i][method[i]];
            std::size_t best = method[i];
            JoinState::Proposal best_proposal;
            double best_gain = kMinImprovement; // strict gains only, so ties cannot cycle
            for (std::size_t c = 0; c < kCandidates.size(); ++c) {
                if (representative[i][c] != c ||
                    representative[i][c] == representative[i][method[i]])
                    continue;
                auto proposal = join.propose(i, views[c][i], state);
                const double gain = separable[i][c] - here + proposal.gain;
                if (gain > best_gain) {
                    best = c;
                    best_gain = gain;
                    best_proposal = std::move(proposal);
                }
            }
            if (best == method[i])
                continue;
            method[i] = best;
            state[i] = views[best][i];
            join.apply(best_proposal);
            ++switches;
        }
        std::cout << std::format("teacher:   pass {}: {} switches, join F1 {:.4f}\n", pass,
                                 switches, join.metric());
        if (switches == 0)
            break;
    }
    return method;
}

// The same workloads with the trajectories in every join pair relabelled at random among
// trajectories that start and end in the same 1 km cells. Each pair keeps its kind and seen flag
// and joins two trajectories that could plausibly match, so best response on them is the same
// search with wrong but believable partners
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
    for (auto *pairs : {&result.positives, &result.negatives})
        for (auto &p : *pairs) {
            const auto a = label[p.i], b = label[p.j];
            p.i = std::min(a, b);
            p.j = std::max(a, b);
            kept_real += pairs == &result.positives && real.contains({p.i, p.j});
        }
    std::cout << std::format("teacher: shuffled partners in {} buckets, {} of {} positive pairs "
                             "still real\n",
                             buckets.size(), kept_real, result.positives.size());
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
    const bool want_original_gate = requested("GATEORIG_a0");
    const bool want_gate = requested("GATE_a0");
    const bool want_rdp_allocation = requested("TEACHER_RDP");
    const bool want_teacher = requested("TEACHER");
    const bool want_equal_response = requested("BR_a0");
    const bool want_response = requested("BR_TEACHER");
    const bool want_shuffled = requested("BR_SHUFFLED");
    const bool want_iterative = requested("ITER_TEACHER");
    const bool needs_allocated_candidates =
        want_teacher || want_response || want_shuffled || want_iterative;
    const bool needs_allocation = want_rdp_allocation || needs_allocated_candidates;
    const bool needs_response_candidates = want_response || want_shuffled || want_iterative;

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
    auto record = [&](std::string_view name, KeptSets kept, const Methods &method, double j_t) {
        std::cout << std::format("teacher {}: {}, J_T {:.4f}\n", name, wins(method), j_t);
        outputs[std::string(name)] = {std::string(name), to_simplification(kept), j_t};
    };
    auto keep = [&](std::string_view name, const Candidates &from, const Methods &method) {
        auto kept = assemble(from, method);
        const double j_t = objective.total(kept);
        record(name, std::move(kept), method, j_t);
    };
    auto start = std::chrono::steady_clock::now();
    auto timed = [&](std::string_view step) {
        std::cout << std::format("teacher: {} {:.1f} s\n", step, seconds_since(start));
        start = std::chrono::steady_clock::now();
    };
    const Methods rdp_methods(db.size(), kRdp);
    const bool equal_alternatives = want_original_gate || want_gate || want_equal_response;
    const auto equal_candidates = make_candidates(objective, equal_budgets, caps,
                                                  equal_alternatives ? nullptr : &rdp_methods);
    timed("candidates");

    if (want_original_gate) {
        keep("GATEORIG_a0", equal_candidates,
             independent_choice(db, objective, equal_candidates, nullptr));
        timed("GATEORIG_a0");
    }

    if (want_gate || want_equal_response || needs_allocation) {
        Curves curves;
        if (needs_allocation) {
            // Partners are fixed at RDP with equal budgets, so trajectories score independently.
            const bool rdp_only = !(want_gate || want_equal_response || needs_allocated_candidates);
            curves = build_curves(db, objective, equal_candidates[kRdp], caps, rdp_only);
            timed(std::format("value curves up to {} points", cap));
        }

        if (want_gate || want_equal_response) {
            // Fixed budgets only need scores at those budgets, not complete value curves.
            const auto gate_methods =
                needs_allocation
                    ? curve_methods(curves.best, equal_budgets)
                    : independent_choice(db, objective, equal_candidates, &equal_candidates[kRdp]);
            if (want_gate)
                keep("GATE_a0", equal_candidates, gate_methods);
            if (want_equal_response) {
                keep("BR_a0", equal_candidates,
                     best_response(db, objective, equal_candidates, gate_methods, m_seed));
                timed("BR_a0");
            }
        }

        auto report = [&](std::string_view what, const AllocationStats &stats,
                          const std::vector<std::size_t> &budgets) {
            std::size_t at_cap = 0;
            for (std::size_t i = 0; i < budgets.size(); ++i)
                at_cap += budgets[i] == caps[i] && caps[i] < db.trajectory(i).size();
            std::cout << std::format("teacher: {}, curve sum {:.4f} -> {:.4f}, {} exchanges, {} "
                                     "trajectories at the cap\n",
                                     what, stats.greedy_value, stats.final_value, stats.exchanges,
                                     at_cap);
        };
        if (want_rdp_allocation) {
            AllocationStats stats;
            const auto budgets = allocate(curves.rdp, total_budget, &stats);
            report("RDP allocation", stats, budgets);
            keep("TEACHER_RDP", make_candidates(objective, budgets, caps, &rdp_methods),
                 rdp_methods);
        }
        if (needs_allocated_candidates) {
            AllocationStats stats;
            const auto budgets = allocate(curves.best, total_budget, &stats);
            report("allocation", stats, budgets);
            const auto teacher_methods = curve_methods(curves.best, budgets);
            const auto allocated_candidates = make_candidates(
                objective, budgets, caps, needs_response_candidates ? nullptr : &teacher_methods);
            if (want_teacher)
                keep("TEACHER", allocated_candidates, teacher_methods);
            if (want_response || want_iterative) {
                const auto response_methods =
                    best_response(db, objective, allocated_candidates, teacher_methods, m_seed);
                timed("BR_TEACHER");
                if (want_response)
                    keep("BR_TEACHER", allocated_candidates, response_methods);
                if (want_iterative) {
                    // Each round's curves see the partners as the last round left them, so the
                    // budgets react to the neighbours too. A round is kept only if J_T rises
                    auto current_kept = assemble(allocated_candidates, response_methods);
                    auto current_methods = response_methods;
                    double current_value = objective.total(current_kept);
                    for (std::size_t round = 1; round <= kMaxAllocationRounds; ++round) {
                        const auto round_curves =
                            build_curves(db, objective, current_kept, caps, false);
                        AllocationStats round_stats;
                        const auto round_budgets =
                            allocate(round_curves.best, total_budget, &round_stats);
                        const auto next_candidates =
                            make_candidates(objective, round_budgets, caps);
                        auto next_methods = best_response(
                            db, objective, next_candidates,
                            curve_methods(round_curves.best, round_budgets), m_seed + round);
                        auto next_kept = assemble(next_candidates, next_methods);
                        const double next_value = objective.total(next_kept);
                        std::cout << std::format("teacher: ITER round {}, J_T {:.4f} -> {:.4f}\n",
                                                 round, current_value, next_value);
                        if (next_value <= current_value + kMinImprovement)
                            break;
                        current_value = next_value;
                        current_kept = std::move(next_kept);
                        current_methods = std::move(next_methods);
                    }
                    record("ITER_TEACHER", std::move(current_kept), current_methods, current_value);
                    timed("ITER_TEACHER");
                }
            }
            if (want_shuffled) {
                // Searches on fake partners; keep scores the result on the real ones
                const auto shuffled = shuffled_partners(db, m_workloads, m_seed);
                const Objective fake(db, shuffled, objective.config());
                keep("BR_SHUFFLED", allocated_candidates,
                     best_response(db, fake, allocated_candidates, teacher_methods, m_seed));
                timed("BR_SHUFFLED");
            }
        }
        timed("allocation arms");
    }

    std::vector<Arm> result;
    result.reserve(arms.size());
    for (const auto &name : arms)
        result.push_back(outputs.at(name));
    return result;
}

} // namespace teacher
