#pragma once

#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/query/join.hpp"

#include <cstddef>
#include <limits>

namespace evaluator {

// Averages over queries. Precision skips queries with an empty simplified result, recall those
// with an empty reference, F1 those where both are empty. NaN if no query counts.
struct JoinScores {
    double f1 = std::numeric_limits<double>::quiet_NaN();
    double precision = std::numeric_limits<double>::quiet_NaN();
    double recall = std::numeric_limits<double>::quiet_NaN();
    double eps_m = std::numeric_limits<double>::quiet_NaN();
    std::size_t query_count = 0;
};


JoinScores evaluate_join(const core::TrajectoryDb &compressed, const query::JoinWorkload &workload);

} // namespace evaluator
