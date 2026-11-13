#pragma once

#include "backend/engine/core/simplification.hpp"
#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/evaluator/evaluator.hpp"
#include "backend/engine/pipeline/config.hpp"
#include "backend/engine/pipeline/test_workloads.hpp"

#include <string_view>

namespace pipeline {

// Scores simplifications of the database on the test workloads, relative to the endpoints-only
// floor, prints the result and exports it if enabled
class Scorer {
  public:
    // Computes and prints the floor. workloads must outlive the scorer.
    Scorer(const core::TrajectoryDb &database, const TestWorkloads &workloads,
           const Config &config);

    // Prints the scores of simplification under label, and exports it as name if enabled
    void score(std::string_view label, std::string_view name,
               const core::Simplification &simplification) const;

  private:
    const core::TrajectoryDb &m_database;
    evaluator::Workloads m_workloads;
    const Config &m_config;
    evaluator::Scoretable m_floor;
};

// Prints "<name>: <n> trajectories, <m> points"
void print_size(std::string_view name, const core::TrajectoryDb &trajectories);

} // namespace pipeline
