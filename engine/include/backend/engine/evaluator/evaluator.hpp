#pragma once

#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/evaluator/where_at.hpp"
#include "backend/engine/query/knn.hpp"
#include "backend/engine/query/range_query.hpp"

#include <limits>
#include <span>

namespace evaluator {

// Kendall's tau is in [-1, 1] and the others in [0, 1], or NaN if it could not be computed
struct Workloads {
    std::span<const query::WhereAtQuery> where_at = {};
    std::span<const query::RangeQuery> range = {};
};

struct Scoretable {
    double knn_kendalls_tau;
    double knn_recall_at_k;
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
    double get_knn_kendalls_tau();
    double get_knn_recall_at_k();
    double get_range_recall();
    WhereAtScores get_where_at_scores();

    // Without a workload the knn scores are NaN. The workload must outlive the evaluator.
    void set_knn_workload(const query::KnnWorkload &workload);

  private:
    // Computes Kendall's tau and recall@k together, as both need the same knn results
    void knn_scores();
    void range_recall();

    const core::TrajectoryDb &m_simplified;
    Workloads m_workloads;
    WhereAtScoringConfig m_where_at_config;

    const query::KnnWorkload *m_knn_workload = nullptr;

    double m_knn_kendalls_tau = std::numeric_limits<double>::quiet_NaN();
    double m_knn_recall_at_k = std::numeric_limits<double>::quiet_NaN();
    double m_range_recall = std::numeric_limits<double>::quiet_NaN();
    WhereAtScores m_where_at_scores;
};

} // namespace evaluator
