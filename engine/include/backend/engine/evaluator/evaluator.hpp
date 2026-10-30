#pragma once

#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/evaluator/knn.hpp"
#include "backend/engine/evaluator/where_at.hpp"
#include "backend/engine/query/knn.hpp"
#include "backend/engine/query/range_query.hpp"

#include <span>

namespace evaluator {

// A missing workload leaves its scores NaN. Workloads must outlive the evaluator.
struct Workloads {
    const query::KnnWorkload *knn = nullptr; // a pointer, as it also carries k and the query db
    std::span<const query::WhereAtQuery> where_at = {};
    std::span<const query::RangeQuery> range = {};
};

// Kendall's tau is in [-1, 1] and the others in [0, 1], or NaN if it could not be computed
struct Scoretable {
    KnnScores knn;
    double range_recall;
    WhereAtScores where_at;
};

struct EvaluationResult {
    Scoretable scores;
    std::vector<KnnAnswer> knn;
    std::vector<std::vector<std::int32_t>> range;
};

class Evaluator {
  public:
    Evaluator(const core::TrajectoryDb &original, const core::TrajectoryDb &simplified,
              Workloads workloads = {}, WhereAtScoringConfig where_at_config = {});

    EvaluationResult evaluate() const;

    // Each runs only its own evaluation
    KnnScores get_knn_scores() const;
    double get_range_recall() const;
    WhereAtScores get_where_at_scores() const;

  private:
    const core::TrajectoryDb &m_simplified;
    Workloads m_workloads;
    WhereAtScoringConfig m_where_at_config;
};

} // namespace evaluator
