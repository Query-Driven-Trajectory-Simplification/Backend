// T1, the independent teacher: per trajectory a value curve V_i(b), the best of RDP, SED and
// UNIFORM at each budget b, then a knapsack over the curves for how many points each trajectory
// gets. The join term sees partners as RDP at equal budgets, a fixed reference, so every
// trajectory is still scored on its own. Arms per share:
//   GATEORIG_a0  equal budgets, partners original (the first, fully independent version)
//   GATE_a0      equal budgets, best curve choice at that budget (selection only)
//   TEACHER_RDP  RDP curves with searched allocation (allocation only)
//   TEACHER      best-of curves with searched allocation (T1)
//   BR_a0        best response from GATE_a0 (T2 step, fixed budgets)
//   BR_TEACHER   best response from TEACHER (T2 step on top of T1)
#include "curves.hpp"
#include "objective.hpp"
#include "teacher.hpp"

#include "backend/engine/allocation/budget_allocator.hpp"
#include "backend/engine/core/parallel.hpp"
#include "backend/engine/query/join.hpp"
#include "backend/engine/selection/selector.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <format>
#include <iostream>
#include <memory>
#include <random>

namespace teacher {

namespace {

using selection::AlgorithmType;
constexpr std::array kCandidates{AlgorithmType::RDP, AlgorithmType::SED, AlgorithmType::UNIFORM};
constexpr std::size_t kRdp = 0;
constexpr std::size_t kMaxPasses = 10;
constexpr std::uint64_t kOrderSeedOffset = 7004;
constexpr std::uint64_t kNegativeSeedOffset = 7005;

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

// candidates[c][i]: trajectory i simplified by kCandidates[c] at budgets[i]
using Candidates = std::array<KeptSets, kCandidates.size()>;

Candidates make_candidates(const core::TrajectoryDb &db, const std::vector<std::size_t> &budgets) {
    Candidates candidates;
    for (auto &kept : candidates)
        kept.resize(db.size());
    core::parallel_for(db.size(), [&](std::size_t i) {
        for (std::size_t c = 0; c < kCandidates.size(); ++c)
            candidates[c][i] =
                selection::simplify_trajectory(db.trajectory(i), kCandidates[c], budgets[i]);
    });
    return candidates;
}

// The best candidate for i against a fixed partner state. Ties go to the earlier candidate
std::size_t best_for(const Objective &objective, const Candidates &candidates, std::size_t i,
                     const KeptSets *partners) {
    std::size_t best = 0;
    double best_value = 0;
    for (std::size_t c = 0; c < kCandidates.size(); ++c) {
        const double v = objective.independent_value(i, candidates[c][i], partners);
        if (c == 0 || v > best_value) {
            best = c;
            best_value = v;
        }
    }
    return best;
}

// Every trajectory picks its best candidate against the same fixed partner state
std::vector<std::size_t> independent_choice(const core::TrajectoryDb &db,
                                            const Objective &objective,
                                            const Candidates &candidates,
                                            const KeptSets *partners) {
    std::vector<std::size_t> method(db.size());
    core::parallel_for(db.size(), [&](std::size_t i) {
        method[i] = best_for(objective, candidates, i, partners);
    });
    return method;
}

KeptSets assemble(const Candidates &candidates, const std::vector<std::size_t> &method) {
    KeptSets kept(method.size());
    for (std::size_t i = 0; i < method.size(); ++i)
        kept[i] = candidates[method[i]][i];
    return kept;
}

std::string wins(const std::vector<std::size_t> &method) {
    std::array<std::size_t, kCandidates.size()> count{};
    for (const auto m : method)
        ++count[m];
    return std::format("RDP {}, SED {}, UNIFORM {}", count[0], count[1], count[2]);
}

// Coordinate ascent on the exact J_T: one trajectory at a time, in a random order, switches to
// the candidate that raises J_T most given everyone's current choice. Only trajectories with join
// candidates can gain, since nothing else depends on the state. Every switch strictly raises
// J_T, so it stops
std::vector<std::size_t> best_response(const core::TrajectoryDb &db, const Objective &objective,
                                       const Candidates &candidates,
                                       std::vector<std::size_t> method, std::uint64_t seed) {
    std::vector<std::size_t> order;
    for (std::size_t i = 0; i < db.size(); ++i)
        if (!objective.links(i).empty())
            order.push_back(i);
    std::mt19937_64 rng(seed + kOrderSeedOffset);
    auto state = assemble(candidates, method);
    JoinState join(objective, state);

    for (std::size_t pass = 1; pass <= kMaxPasses; ++pass) {
        std::ranges::shuffle(order, rng);
        std::size_t switches = 0;
        for (const auto i : order) {
            const double here = objective.separable(i, state[i]);
            std::size_t best = method[i];
            double best_gain = 1e-12; // strict gain only, so ties cannot cycle
            for (std::size_t c = 0; c < kCandidates.size(); ++c) {
                if (c == method[i])
                    continue;
                const double gain = objective.separable(i, candidates[c][i]) - here +
                                    join.delta(i, candidates[c][i], state);
                if (gain > best_gain) {
                    best = c;
                    best_gain = gain;
                }
            }
            if (best == method[i])
                continue;
            method[i] = best;
            state[i] = candidates[best][i];
            join.apply(i, state);
            ++switches;
        }
        std::cerr << std::format("teacher:   pass {}: {} switches, J_T {:.4f}, join F1 {:.4f} "
                                 "(tracked {:.4f})\n",
                                 pass, switches, objective.total(state),
                                 objective.metrics(state)[Objective::Join], join.metric());
        if (switches == 0)
            break;
    }
    return method;
}

// Curves up to cap points (and never more than a trajectory has): the best candidate's value,
// and RDP's alone. Scored independently, partners fixed at the given reference state
struct Curves {
    std::vector<Curve> best, rdp;
};

Curves build_curves(const core::TrajectoryDb &db, const Objective &objective,
                    const KeptSets &partners, std::size_t cap) {
    Curves curves{std::vector<Curve>(db.size()), std::vector<Curve>(db.size())};
    core::parallel_for(db.size(), [&](std::size_t i) {
        const auto trajectory = db.trajectory(i);
        const std::size_t max_b = std::min(cap, trajectory.size());
        auto &best = curves.best[i];
        auto &rdp = curves.rdp[i];
        for (std::size_t b = 2; b <= max_b; ++b) {
            double best_value = 0;
            std::uint8_t best_method = 0;
            for (std::size_t c = 0; c < kCandidates.size(); ++c) {
                // At two points every candidate keeps the endpoints
                if (b == 2 && c > 0)
                    break;
                const auto kept = selection::simplify_trajectory(trajectory, kCandidates[c], b);
                const double v = objective.independent_value(i, kept, &partners);
                if (c == 0)
                    rdp.value.push_back(v);
                if (c == 0 || v > best_value) {
                    best_value = v;
                    best_method = static_cast<std::uint8_t>(c);
                }
            }
            best.value.push_back(best_value);
            best.method.push_back(best_method);
            rdp.method.push_back(0);
        }
    });
    return curves;
}

// Each trajectory's curve method at its budget
std::vector<std::size_t> curve_methods(const std::vector<Curve> &curves,
                                       const std::vector<std::size_t> &budgets) {
    std::vector<std::size_t> method(curves.size());
    for (std::size_t i = 0; i < curves.size(); ++i) {
        if (budgets[i] < 2 || budgets[i] > curves[i].max_budget())
            throw std::runtime_error("teacher: budget outside its curve");
        method[i] = curves[i].method[budgets[i] - 2];
    }
    return method;
}

// Candidate pairs (endpoints within 150 m) whose originals are not within 150 m, half seen
std::vector<JoinPair> negative_pairs(const core::TrajectoryDb &db, std::uint64_t seed) {
    const auto join = query::generate_self_join(db, kJoinEps, query::FrechetVariant::Continuous);
    std::vector<JoinPair> negatives;
    for (const auto &entry : join.entries)
        for (const auto j : entry.candidates)
            if (!std::ranges::binary_search(entry.reference, j))
                negatives.push_back({static_cast<std::uint32_t>(entry.query_index),
                                     static_cast<std::uint32_t>(j), false});
    std::mt19937_64 rng(seed + kNegativeSeedOffset);
    std::ranges::shuffle(negatives, rng);
    for (std::size_t k = 0; k < negatives.size() / 2; ++k)
        negatives[k].seen = true;
    return negatives;
}

} // namespace

TeacherOutput run_teacher(const core::TrajectoryDb &db, const TeacherCaches &caches,
                          std::uint64_t seed, std::span<const double> shares) {
    TeacherOutput output;

    auto start = std::chrono::steady_clock::now();
    const auto negatives = negative_pairs(db, seed);
    output.timings_s["negative_pairs"] = seconds_since(start);
    start = std::chrono::steady_clock::now();
    auto objective = std::make_shared<const Objective>(db, caches, negatives);
    output.timings_s["objective_setup"] = seconds_since(start);
    std::cerr << std::format("teacher: {} seen join pairs, {} positive, {} negative\n",
                             objective->pair_count(), objective->positive_count(),
                             objective->negative_count());
    const auto &floor = objective->floors();
    std::cerr << std::format("teacher: floors knn {:.4f}, range {:.4f}, where-at {:.4f}, join "
                             "{:.4f}\n",
                             floor[Objective::Knn], floor[Objective::Range],
                             floor[Objective::WhereAt], floor[Objective::Join]);

    for (const double share : shares) {
        const auto tag = share_tag(share);
        const auto budget = total_budget(db, share);
        const auto equal = allocation::create_trajectory_budgets(
            db, allocation::AllocatorType::SizeProportional, budget, 0.0);
        start = std::chrono::steady_clock::now();
        const auto candidates = make_candidates(db, equal);
        output.timings_s["candidates_" + tag] = seconds_since(start);

        auto emit = [&](const std::string &name, const Candidates &from,
                        const std::vector<std::size_t> &method) {
            auto kept = assemble(from, method);
            std::cerr << std::format("teacher {} {}: {}, J_T {:.4f}\n", tag, name, wins(method),
                                     objective->total(kept));
            output.arms[name + "_" + tag] = std::move(kept);
        };

        start = std::chrono::steady_clock::now();
        emit("GATEORIG_a0", candidates, independent_choice(db, *objective, candidates, nullptr));
        output.timings_s["gate_orig_" + tag] = seconds_since(start);

        // Curves up to about four times the mean interior budget, at least 6 interior points
        const double mean_interior =
            static_cast<double>(budget - 2 * db.size()) / static_cast<double>(db.size());
        const auto cap = 2 + static_cast<std::size_t>(std::ceil(std::max(6.0, 4.0 * mean_interior)));
        start = std::chrono::steady_clock::now();
        const auto curves = build_curves(db, *objective, candidates[kRdp], cap);
        output.timings_s["curves_" + tag] = seconds_since(start);
        std::cerr << std::format("teacher {}: curves up to {} points\n", tag, cap);

        const auto gate = curve_methods(curves.best, equal);
        emit("GATE_a0", candidates, gate);

        start = std::chrono::steady_clock::now();
        AllocationStats rdp_stats, best_stats;
        const auto rdp_budgets = allocate(curves.rdp, budget, &rdp_stats);
        const auto teacher_budgets = allocate(curves.best, budget, &best_stats);
        output.timings_s["allocation_" + tag] = seconds_since(start);
        std::cerr << std::format("teacher {}: allocation curve sum RDP {:.4f} -> {:.4f} ({} "
                                 "exchanges), best {:.4f} -> {:.4f} ({} exchanges)\n",
                                 tag, rdp_stats.greedy_value, rdp_stats.final_value,
                                 rdp_stats.exchanges, best_stats.greedy_value,
                                 best_stats.final_value, best_stats.exchanges);

        const auto rdp_candidates = make_candidates(db, rdp_budgets);
        emit("TEACHER_RDP", rdp_candidates, std::vector<std::size_t>(db.size(), kRdp));
        const auto teacher_candidates = make_candidates(db, teacher_budgets);
        const auto teacher = curve_methods(curves.best, teacher_budgets);
        emit("TEACHER", teacher_candidates, teacher);

        start = std::chrono::steady_clock::now();
        emit("BR_a0", candidates, best_response(db, *objective, candidates, gate, seed));
        output.timings_s["best_response_a0_" + tag] = seconds_since(start);
        start = std::chrono::steady_clock::now();
        emit("BR_TEACHER", teacher_candidates,
             best_response(db, *objective, teacher_candidates, teacher, seed));
        output.timings_s["best_response_teacher_" + tag] = seconds_since(start);
    }

    // Keeps the objective alive for the harness's J_T column
    output.objective = [objective](const KeptSets &kept, double) { return objective->total(kept); };
    return output;
}

} // namespace teacher
