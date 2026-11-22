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

//   BUDGET_RDP         budgets from value curves, RDP
//   BUDGET_CHOICE      budgets from value curves, best candidate each
//   BUDGET_LOOP        then best response against the others' current state until nothing switches
//   BUDGET_LOOP_TRADE  then moves points between trajectories while exact J_T rises: the teacher
inline constexpr std::array<std::string_view, 4> kArms{"BUDGET_RDP", "BUDGET_CHOICE", "BUDGET_LOOP",
                                                       "BUDGET_LOOP_TRADE"};

bool is_arm(std::string_view name);
// Throws std::invalid_argument on an unknown or repeated arm
void check_arms(std::span<const std::string> arms);

struct TeacherConfig {
    bool enabled = false;
    std::filesystem::path cache_dir = "cache";
    std::vector<std::string> arms{"BUDGET_CHOICE", "BUDGET_LOOP"};
    std::size_t range_query_count = kRangeQueryCount;
    // Unset: join.score_eps_m, the thresholds J4 averages over
    std::optional<std::vector<double>> extra_join_eps_m;
    TermWeights term_weights{1.0, 1.0, 1.0, 1.0, 0.0, 0.0};
};

struct Arm {
    std::string name;
    core::Simplification simplification;
    double objective = 0.0; // J_T
};

// Sees only the database it is given and workloads built from it. Trained for held-out queries
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
