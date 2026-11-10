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

// The arms a teacher run can produce. Selection only: every trajectory keeps its equal share of
// the budget (alpha = 0) and the teacher picks RDP, SED or UNIFORM for it:
//   GATEORIG_a0  join partners seen as original (fully independent)
//   GATE_a0      the best choice on each trajectory's value curve, partners seen as RDP
//   BR_a0        best response from GATE_a0: re-pick against partners' current choice (T2 step)
inline constexpr std::array<std::string_view, 3> kArms{"GATEORIG_a0", "GATE_a0", "BR_a0"};

bool is_arm(std::string_view name);

struct TeacherConfig {
    bool enabled = false;
    std::filesystem::path cache_dir = "cache"; // looked in first, written to after a build
    std::vector<std::string> arms{"GATE_a0", "BR_a0"};
    std::size_t range_query_count = 64000; // the teacher's own boxes, denser than the test's
};

struct Arm {
    std::string name;
    core::Simplification simplification;
    double objective = 0.0; // J_T on the teacher's workloads
};

// Sees only the database it is given and workloads built from it
class Teacher {
  public:
    Teacher(const core::TrajectoryDb &db, TeacherWorkloads workloads,
            const ObjectiveConfig &config, std::uint64_t seed);
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
