#pragma once

#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/query/knn.hpp"

#include <cstddef>
#include <limits>
#include <vector>

namespace evaluator {

struct KnnScores {
    double kendalls_tau = std::numeric_limits<double>::quiet_NaN();
    double recall_at_k = std::numeric_limits<double>::quiet_NaN();
    std::size_t query_count = 0;
    std::size_t tau_query_count = 0; // queries with at least two shared neighbors
};

// Each score is averaged over the queries where it is defined, NaN if it is defined for none.
KnnScores evaluate_knn(const core::TrajectoryDb &compressed, const query::KnnWorkload &workload);

} // namespace evaluator

namespace evaluator::knn {

// Rank agreement on shared neighbors only; NaN when fewer than two are shared.
double kendalls_tau(const std::vector<std::size_t> &original,
                    const std::vector<std::size_t> &simplified);

// Share of original that also appears in simplified, ignoring order. NaN if original is empty.
double recall_at_k(const std::vector<std::size_t> &original,
                   const std::vector<std::size_t> &simplified);

} // namespace evaluator::knn
