#pragma once

#include "backend/engine/core/simplification.hpp"
#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/teacher/objective.hpp"
#include "backend/engine/teacher/workloads.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace teacher {

// The arms a teacher run can produce. All pick a candidate per trajectory (RDP, SED, UNIFORM or
// one of the greedy variants):
//   GATEORIG_a0  equal budgets, join partners seen as original (fully independent)
//   GATE_a0      equal budgets, best candidate against RDP partners (selection only)
//   TEACHER_RDP  RDP with allocation searched on its value curves (allocation only)
//   TEACHER      best-of value curves with searched allocation (T1)
//   BR_a0        best response from GATE_a0: re-pick against partners' current choice (T2 step)
//   BR_TEACHER   best response from TEACHER
//   BR_SHUFFLED  the same search from TEACHER, partners relabelled among nearby trajectories
//                (control)
//   ITER_TEACHER BR_TEACHER, then rounds of value curves against the partners' current state,
//                allocation and best response, as long as J_T rises
inline constexpr std::array<std::string_view, 8> kArms{"GATEORIG_a0", "GATE_a0",     "TEACHER_RDP",
                                                       "TEACHER",     "BR_a0",       "BR_TEACHER",
                                                       "BR_SHUFFLED", "ITER_TEACHER"};

bool is_arm(std::string_view name);

struct TeacherConfig {
    bool enabled = false;
    std::filesystem::path cache_dir = "cache"; // looked in first, written to after a build
    std::vector<std::string> arms{"TEACHER", "BR_TEACHER"};
    std::size_t range_query_count = 64000; // the teacher's own boxes, denser than the test's
    bool all_join_pairs = true;            // false: only the seen half of the join pairs
    double shape_weight = 0.0;             // the objective's shape term, 0 = off
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

    const Objective &objective() const { return m_objective; }

  private:
    const core::TrajectoryDb &m_db;
    TeacherWorkloads m_workloads; // before m_objective, which refers to it
    Objective m_objective;
    std::uint64_t m_seed;
};

} // namespace teacher
