#pragma once

#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/query/knn.hpp"

#include <cstddef>
#include <limits>

namespace evaluator {

// Kendall's tau is in [-1, 1] and the others in [0, 1], or NaN if it could not be computed
struct Scoretable {
    double knn_kendalls_tau;
    double knn_recall_at_k;
    double range_score;
    double where_at_score;
};

class Evaluator {
  public:
    Evaluator(const core::TrajectoryDb &original, const core::TrajectoryDb &simplified);

    // Runs every evaluation, stores the scores and returns them
    Scoretable evaluate();

    // Each runs only its own evaluation, stores the score and returns it
    double get_knn_kendalls_tau();
    double get_knn_recall_at_k();
    double get_range_score();
    double get_where_at_score();

    // Without a workload the knn scores are NaN. The workload must outlive the evaluator.
    void set_knn_workload(const query::KnnWorkload &workload);

  private:
    // Computes Kendall's tau and recall@k together, as both need the same knn results
    void knn_scores();
    void range_score();
    void where_at_score();

    const core::TrajectoryDb &m_original;
    const core::TrajectoryDb &m_simplified;

    const query::KnnWorkload *m_knn_workload = nullptr;

    double m_knn_kendalls_tau = std::numeric_limits<double>::quiet_NaN();
    double m_knn_recall_at_k = std::numeric_limits<double>::quiet_NaN();
    double m_range_score = std::numeric_limits<double>::quiet_NaN();
    double m_where_at_score = std::numeric_limits<double>::quiet_NaN();
};

} // namespace evaluator
