#pragma once

#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/query/join.hpp"

#include <cstddef>
#include <limits>

namespace evaluator {

// HeldOut mode averages over queries. Self mode uses global unordered-pair micro scores.
// Self precision/recall are NaN for empty prediction/reference respectively; F1 is NaN
// only when both pair sets are empty, otherwise 2*TP/(predicted+reference).
// HeldOut: averages over queries. Precision skips queries with an empty simplified result, recall
// those with an empty reference, F1 those where both are empty. NaN if no query counts.
struct JoinScores {
    double f1 = std::numeric_limits<double>::quiet_NaN();
    double precision = std::numeric_limits<double>::quiet_NaN();
    double recall = std::numeric_limits<double>::quiet_NaN();
    double eps_m = std::numeric_limits<double>::quiet_NaN();
    std::size_t query_count = 0; // held-out queries only; zero for Self
    query::JoinMode mode = query::JoinMode::HeldOut;
    std::size_t candidate_pairs = 0;
    std::size_t reference_pairs = 0;
    std::size_t predicted_pairs = 0;
    std::size_t true_positive_pairs = 0;
};

JoinScores evaluate_join(const core::TrajectoryDb &compressed, const query::JoinWorkload &workload);

} // namespace evaluator
