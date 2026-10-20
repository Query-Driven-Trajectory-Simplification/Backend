#pragma once

#include "backend/engine/core/trajectory_db.hpp"

#include <cstddef>

namespace evaluator {

struct Scoretable {
    double knn_score;
    // add where at, range score, etc..
};

// Scores how well simplified answers the queries compared to original. simplified must hold the
// same trajectories in the same order as original, e.g. from core::materialize.
Scoretable evaluate_simplification(const core::TrajectoryDb &original,
                                   const core::TrajectoryDb &simplified,
                                   const core::TrajectoryDb &queries, std::size_t k);

} // namespace evaluator
