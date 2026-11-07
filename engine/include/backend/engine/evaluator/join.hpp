#pragma once

#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/query/join.hpp"

#include <cstddef>
#include <limits>

namespace evaluator {

// Averages over queries, so every trajectory weighs the same however many partners it has.
// In Self mode every database trajectory is a query, and a pair counts for both of its
// trajectories. Precision skips queries with an empty simplified result, recall those with an
// empty reference, F1 those where both are empty. NaN if no query counts.
// The micro scores pool all pairs instead, a diagnostic that busy routes dominate.
struct JoinScores {
    double f1 = std::numeric_limits<double>::quiet_NaN();
    double precision = std::numeric_limits<double>::quiet_NaN();
    double recall = std::numeric_limits<double>::quiet_NaN();
    double micro_f1 = std::numeric_limits<double>::quiet_NaN();
    double micro_precision = std::numeric_limits<double>::quiet_NaN();
    double micro_recall = std::numeric_limits<double>::quiet_NaN();
    double eps_m = std::numeric_limits<double>::quiet_NaN();
    std::size_t query_count = 0; // Self: database trajectories
    query::JoinMode mode = query::JoinMode::HeldOut;
    std::size_t candidate_pairs = 0;
    std::size_t reference_pairs = 0;
    std::size_t predicted_pairs = 0;
    std::size_t true_positive_pairs = 0;
};

JoinScores evaluate_join(const core::TrajectoryDb &compressed, const query::JoinWorkload &workload);

} // namespace evaluator
