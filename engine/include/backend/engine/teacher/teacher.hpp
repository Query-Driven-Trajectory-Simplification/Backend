#pragma once

#include "backend/engine/core/simplification.hpp"
#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/teacher/objective.hpp"
#include "backend/engine/teacher/workloads.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace teacher {

// The arms a teacher run can produce. Each keeps one candidate per trajectory (RDP, SED, UNIFORM
// or one of the greedy variants); "loop" is best response against the neighbours' current
// compressed state, repeated until no trajectory switches:
//   EQUAL_CHOICE              equal budgets, each trajectory's best candidate on its own, join
//                             partners seen uncompressed
//   EQUAL_CHOICE_RDP_PARTNERS the same with join partners guessed as RDP at equal budgets
//   EQUAL_LOOP                the loop from EQUAL_CHOICE_RDP_PARTNERS
//   BUDGET_RDP                budgets searched on value curves, RDP's points
//   BUDGET_CHOICE             budgets searched on value curves, each trajectory's best candidate
//   BUDGET_LOOP               the loop from BUDGET_CHOICE: the teacher
//   BUDGET_LOOP_REALLOCATE    BUDGET_LOOP, then rounds of value curves against the current state,
//                             budget search and the loop again, as long as J_T rises, so budgets
//                             react to the neighbours too
//   BUDGET_LOOP_SHUFFLED      the same loop with join partners and kNN neighbours relabelled
//                             among nearby trajectories, scored on the real ones (control)
//   BUDGET_LOOP_NO_UPDATE     one pass from BUDGET_CHOICE in which every trajectory picks its best
//                             candidate against the others' starting state, without seeing their
//                             switches (control: how much reacting adds to looking once)
inline constexpr std::array<std::string_view, 9> kArms{"EQUAL_CHOICE",
                                                       "EQUAL_CHOICE_RDP_PARTNERS",
                                                       "EQUAL_LOOP",
                                                       "BUDGET_RDP",
                                                       "BUDGET_CHOICE",
                                                       "BUDGET_LOOP",
                                                       "BUDGET_LOOP_REALLOCATE",
                                                       "BUDGET_LOOP_SHUFFLED",
                                                       "BUDGET_LOOP_NO_UPDATE"};

bool is_arm(std::string_view name);

struct TeacherConfig {
    bool enabled = false;
    std::filesystem::path cache_dir = "cache"; // looked in first, written to after a build
    std::vector<std::string> arms{"BUDGET_CHOICE", "BUDGET_LOOP"};
    std::size_t range_query_count = 64000; // the teacher's own boxes, denser than the test's
    bool all_join_pairs = true;            // false: only the seen half of the join pairs
    // The objective's join term (ObjectiveConfig::join_mode). Unset: the join.mode it is scored on
    std::optional<query::JoinMode> join;
    std::vector<double> extra_join_eps_m; // more self-join thresholds for the join term
    bool held_out_term = false;           // ObjectiveConfig::held_out_term
    bool ranking_knn = true;              // ObjectiveConfig::ranking_knn
    // ObjectiveConfig::term_weights: kNN, range, where-at, join
    std::array<double, 4> term_weights{1.0, 1.0, 1.0, 1.0};
};

struct Arm {
    std::string name;
    core::Simplification simplification;
    double objective = 0.0; // J_T on the teacher's workloads
};

// Sees only the database it is given and workloads built from it
class Teacher {
  public:
    Teacher(const core::TrajectoryDb &db, TeacherWorkloads workloads, const ObjectiveConfig &config,
            std::uint64_t seed);
    Teacher(const Teacher &) = delete;
    Teacher &operator=(const Teacher &) = delete;

    // The requested arms, in order, each keeping exactly total_budget points
    std::vector<Arm> run(std::size_t total_budget, std::span<const std::string> arms) const;

    // Each candidate recipe for every trajectory at equal budgets, named RECIPE_<recipe>: the
    // single-recipe baselines that choosing per trajectory is compared against
    std::vector<Arm> single_recipes(std::size_t total_budget) const;

    const Objective &objective() const { return m_objective; }

  private:
    const core::TrajectoryDb &m_db;
    TeacherWorkloads m_workloads; // before m_objective, which refers to it
    Objective m_objective;
    std::uint64_t m_seed;
};

} // namespace teacher
