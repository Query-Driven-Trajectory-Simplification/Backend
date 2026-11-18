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

// "Loop" is best response against the neighbours' current state until nothing switches.
//   EQUAL_CHOICE              equal budgets, best candidate each, partners original
//   EQUAL_CHOICE_RDP_PARTNERS the same, partners RDP
//   EQUAL_LOOP                the loop from EQUAL_CHOICE_RDP_PARTNERS
//   BUDGET_RDP                budgets from value curves, RDP
//   BUDGET_CHOICE             budgets from value curves, best candidate each
//   BUDGET_LOOP               the loop from BUDGET_CHOICE: the teacher
//   BUDGET_LOOP_REALLOCATE    then curves, budgets and loop again while J_T rises. With only
//                             held-out join terms the curves ignore partners, so this only reruns
//                             the loop in a new visit order
//   BUDGET_LOOP_SHUFFLED      control: loop on relabelled partners and neighbours
//   BUDGET_LOOP_NO_UPDATE     control: one pass against the starting state
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
    std::filesystem::path cache_dir = "cache";
    std::vector<std::string> arms{"BUDGET_CHOICE", "BUDGET_LOOP"};
    std::size_t range_query_count = 64000;
    bool all_join_pairs = true; // false: only the seen half of the join pairs
    // Unset: join.mode
    std::optional<query::JoinMode> join;
    std::vector<double> extra_join_eps_m;
    bool held_out_term = false;
    bool ranking_knn = true;
    std::array<double, 4> term_weights{1.0, 1.0, 1.0, 1.0};
};

struct Arm {
    std::string name;
    core::Simplification simplification;
    double objective = 0.0; // J_T
};

// Sees only the database it is given and workloads built from it
class Teacher {
  public:
    Teacher(const core::TrajectoryDb &db, TeacherWorkloads workloads, const ObjectiveConfig &config,
            std::uint64_t seed);
    Teacher(const Teacher &) = delete;
    Teacher &operator=(const Teacher &) = delete;

    // The requested arms, in order
    std::vector<Arm> run(std::size_t total_budget, std::span<const std::string> arms) const;

    // RECIPE_<recipe>: one recipe for every trajectory at equal budgets
    std::vector<Arm> single_recipes(std::size_t total_budget) const;

    const Objective &objective() const { return m_objective; }

  private:
    const core::TrajectoryDb &m_db;
    TeacherWorkloads m_workloads; // before m_objective, which refers to it
    Objective m_objective;
    std::uint64_t m_seed;
};

} // namespace teacher
