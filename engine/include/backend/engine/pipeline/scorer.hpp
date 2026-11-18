#pragma once

#include "backend/engine/core/simplification.hpp"
#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/evaluator/evaluator.hpp"
#include "backend/engine/pipeline/config.hpp"
#include "backend/engine/pipeline/test_workloads.hpp"

#include <string_view>

namespace pipeline {

// Scores simplifications against the endpoints-only floor, prints and optionally exports them
class Scorer {
  public:
    // workloads must outlive the scorer
    Scorer(const core::TrajectoryDb &database, const TestWorkloads &workloads,
           const Config &config);

    void score(std::string_view label, std::string_view name,
               const core::Simplification &simplification) const;

  private:
    const core::TrajectoryDb &m_database;
    evaluator::Workloads m_workloads;
    const Config &m_config;
    evaluator::Scoretable m_floor;
};

void print_size(std::string_view name, const core::TrajectoryDb &trajectories);

} // namespace pipeline
