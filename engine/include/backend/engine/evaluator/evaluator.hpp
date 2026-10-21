#pragma once

#include "backend/engine/core/trajectory_db.hpp"

#include <cstddef>

namespace evaluator {

struct Scoretable {
    double knn_score;
    // add where at, range score, etc..
};

// Scores how well simplified answers queries compared to original, using queries from
// query::generate_knn_queries. simplified must hold the same trajectories in the same order as
// original, e.g. from core::materialize.
Scoretable evaluate_simplification(const core::TrajectoryDb &original,
                                   const core::TrajectoryDb &simplified);

} // namespace evaluator
