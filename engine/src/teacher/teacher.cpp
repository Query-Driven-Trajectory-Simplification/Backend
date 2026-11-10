#include "backend/engine/teacher/teacher.hpp"

#include "backend/engine/allocation/budget_allocator.hpp"
#include "backend/engine/core/parallel.hpp"
#include "backend/engine/selection/selector.hpp"
#include "backend/engine/teacher/curves.hpp"

#include <algorithm>
#include <chrono>
#include <format>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>

namespace teacher {

namespace {

using selection::AlgorithmType;
constexpr std::array kCandidates{AlgorithmType::RDP, AlgorithmType::SED, AlgorithmType::UNIFORM};
constexpr std::size_t kRdp = 0;
constexpr std::size_t kMaxPasses = 10;
constexpr std::uint64_t kOrderSeedOffset = 7004;

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

// candidates[c][i]: trajectory i simplified by kCandidates[c] at budgets[i]
using Candidates = std::array<KeptSets, kCandidates.size()>;
using Methods = std::vector<std::size_t>; // index into kCandidates, per trajectory

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

KeptSets assemble(const Candidates &candidates, const Methods &method) {
    KeptSets kept(method.size());
    for (std::size_t i = 0; i < method.size(); ++i)
        kept[i] = candidates[method[i]][i];
    return kept;
}

// Every trajectory picks its best candidate against the same fixed partner state. Ties go to the
// earlier candidate, so RDP wins when nothing differs
Methods independent_choice(const core::TrajectoryDb &db, const Objective &objective,
                           const Candidates &candidates, const KeptSets *partners) {
    Methods method(db.size());
    core::parallel_for(db.size(), [&](std::size_t i) {
        double best = 0;
        for (std::size_t c = 0; c < kCandidates.size(); ++c) {
            const double v = objective.independent_value(i, candidates[c][i], partners);
            if (c == 0 || v > best) {
                best = v;
                method[i] = c;
            }
        }
    });
    return method;
}

// Value curves from 2 up to max_budget[i] points: the best candidate's value and method at each
// budget. Scored independently, partners fixed at the given reference state
std::vector<Curve> build_curves(const core::TrajectoryDb &db, const Objective &objective,
                                const KeptSets &partners,
                                const std::vector<std::size_t> &max_budget) {
    std::vector<Curve> curves(db.size());
    core::parallel_for(db.size(), [&](std::size_t i) {
        const auto trajectory = db.trajectory(i);
        const std::size_t max_b = std::min(max_budget[i], trajectory.size());
        auto &best = curves[i];
        for (std::size_t b = 2; b <= max_b; ++b) {
            double best_value = 0;
            std::uint8_t best_method = 0;
            for (std::size_t c = 0; c < kCandidates.size(); ++c) {
                // At two points every candidate keeps the endpoints
                if (b == 2 && c > 0)
                    break;
                const auto kept = selection::simplify_trajectory(trajectory, kCandidates[c], b);
                const double v = objective.independent_value(i, kept, &partners);
                if (c == 0 || v > best_value) {
                    best_value = v;
                    best_method = static_cast<std::uint8_t>(c);
                }
            }
            best.value.push_back(best_value);
            best.method.push_back(best_method);
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
    std::mt19937_64 rng(seed + kOrderSeedOffset);
    auto state = assemble(candidates, method);
    JoinState join(objective, state);

    for (std::size_t pass = 1; pass <= kMaxPasses; ++pass) {
        std::ranges::shuffle(order, rng);
        std::size_t switches = 0;
        for (const auto i : order) {
            const double here = objective.separable(i, state[i]);
            std::size_t best = method[i];
            double best_gain = 1e-12; // strict gains only, so ties cannot cycle
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
        std::cout << std::format("teacher:   pass {}: {} switches, join F1 {:.4f}\n", pass,
                                 switches, join.metric());
        if (switches == 0)
            break;
    }
    return method;
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
    return std::format("RDP {}, SED {}, UNIFORM {}", count[0], count[1], count[2]);
}

bool wants(std::span<const std::string> arms, std::initializer_list<std::string_view> any) {
    return std::ranges::any_of(arms, [&](const std::string &a) {
        return std::ranges::find(any, std::string_view(a)) != any.end();
    });
}

} // namespace

bool is_arm(std::string_view name) { return std::ranges::find(kArms, name) != kArms.end(); }

Teacher::Teacher(const core::TrajectoryDb &db, TeacherWorkloads workloads,
                 const ObjectiveConfig &config, std::uint64_t seed)
    : m_db(db), m_workloads(std::move(workloads)), m_objective(db, m_workloads, config),
      m_seed(seed) {
    const auto &f = m_objective.floors();
    std::cout << std::format("teacher: {} seen join pairs ({} positive), floors knn {:.4f}, range "
                             "{:.4f}, where-at {:.4f}, join {:.4f}\n",
                             m_objective.pair_count(), m_objective.positive_count(),
                             f[Objective::Knn], f[Objective::Range], f[Objective::WhereAt],
                             f[Objective::Join]);
}

std::vector<Arm> Teacher::run(std::size_t total_budget, std::span<const std::string> arms) const {
    for (const auto &a : arms)
        if (!is_arm(a))
            throw std::invalid_argument("teacher: unknown arm " + a);
    const auto &db = m_db;
    const auto &objective = m_objective;
    const auto equal = allocation::create_trajectory_budgets(
        db, allocation::AllocatorType::SizeProportional, total_budget, 0.0);
    const auto candidates = make_candidates(db, equal);

    std::map<std::string, std::pair<KeptSets, double>, std::less<>> made; // kept sets, J_T
    auto keep = [&](std::string_view name, const Candidates &from, const Methods &method) {
        auto kept = assemble(from, method);
        const double j_t = objective.total(kept);
        std::cout << std::format("teacher {}: {}, J_T {:.4f}\n", name, wins(method), j_t);
        made[std::string(name)] = {std::move(kept), j_t};
    };
    auto start = std::chrono::steady_clock::now();
    auto timed = [&](std::string_view step) {
        std::cout << std::format("teacher: {} {:.1f} s\n", step, seconds_since(start));
        start = std::chrono::steady_clock::now();
    };

    if (wants(arms, {"GATEORIG_a0"})) {
        keep("GATEORIG_a0", candidates, independent_choice(db, objective, candidates, nullptr));
        timed("GATEORIG_a0");
    }

    if (wants(arms, {"GATE_a0", "BR_a0"})) {
        // Partners are fixed at RDP with equal budgets, so each trajectory is scored on its own.
        // The curves run up to each trajectory's budget; selection reads them there
        const auto curves = build_curves(db, objective, candidates[kRdp], equal);
        timed("value curves");

        const auto gate = curve_methods(curves, equal);
        if (wants(arms, {"GATE_a0"}))
            keep("GATE_a0", candidates, gate);
        if (wants(arms, {"BR_a0"})) {
            keep("BR_a0", candidates, best_response(db, objective, candidates, gate, m_seed));
            timed("BR_a0");
        }
    }

    std::vector<Arm> result;
    for (const auto &name : arms) {
        const auto &[kept, j_t] = made.at(name);
        result.push_back({name, to_simplification(kept), j_t});
    }
    return result;
}

} // namespace teacher
