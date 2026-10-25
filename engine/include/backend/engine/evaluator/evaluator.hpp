#pragma once

#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/evaluator/knn_evaluation.hpp"
#include "backend/engine/evaluator/where_at.hpp"
#include "backend/engine/query/knn.hpp"
#include "backend/engine/query/range_query.hpp"

#include <limits>
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

class Evaluator {
  public:
    Evaluator(const core::TrajectoryDb &original, const core::TrajectoryDb &simplified,
              Workloads workloads = {}, WhereAtScoringConfig where_at_config = {});

    // Runs every evaluation, stores the scores and returns them
    Scoretable evaluate();

    // Each runs only its own evaluation, stores the score and returns it
    KnnScores get_knn_scores();
    double get_range_recall();
    WhereAtScores get_where_at_scores();

  private:
    const core::TrajectoryDb &m_simplified;
    Workloads m_workloads;
    WhereAtScoringConfig m_where_at_config;

    KnnScores m_knn_scores;
    double m_range_recall = std::numeric_limits<double>::quiet_NaN();
    WhereAtScores m_where_at_scores;
};

} // namespace evaluator
