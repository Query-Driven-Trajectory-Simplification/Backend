#pragma once

#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/query/knn.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace evaluator {

struct KnnScores {
    double recall_at_k = std::numeric_limits<double>::quiet_NaN();
    double kendalls_tau_topk = std::numeric_limits<double>::quiet_NaN();
    std::size_t query_count = 0;
};

struct KnnAnswer {
    std::int32_t query_trajectory_id;
    std::vector<std::int32_t> neighbors;
};

struct KnnEvaluation {
    KnnScores scores;
    std::vector<KnnAnswer> answers;
};

// Executes each query once and returns both scores and answers in workload entry order.
KnnEvaluation evaluate_knn(const core::TrajectoryDb &compressed,
                           const query::KnnWorkload &workload);

} // namespace evaluator

namespace evaluator::knn {

// Fagin's K^(p) over the union of both lists as a similarity in [0, 1], 1 = identical. A pair
// that one list holds neither item of costs p. NaN if no pair can be penalized (e.g. both empty).
double kendalls_tau_topk(const std::vector<std::size_t> &original,
                         const std::vector<std::size_t> &simplified, double p);

// Share of original that also appears in simplified, ignoring order. NaN if original is empty.
double recall_at_k(const std::vector<std::size_t> &original,
                   const std::vector<std::size_t> &simplified);

} // namespace evaluator::knn
