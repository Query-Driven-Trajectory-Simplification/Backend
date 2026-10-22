#pragma once

#include "backend/engine/core/trajectory_db.hpp"

#include <cstddef>
#include <limits>

namespace evaluator {

// Each score is in [-1, 1] for knn and [0, 1] for the others, or NaN if it could not be computed
struct Scoretable {
    double knn_score;
    double range_score;
    double where_at_score;
};

class Evaluator {
  public:
    Evaluator(const core::TrajectoryDb &original, const core::TrajectoryDb &simplified);

    // Runs every evaluation, stores the scores and returns them
    Scoretable evaluate();

    // Each runs only its own evaluation, stores the score and returns it
    double get_knn_score();
    double get_range_score();
    double get_where_at_score();

    void set_knn_neighbours(std::size_t k);

  private:

    void knn_score(std::size_t k);
    void range_score();
    void where_at_score();

    const core::TrajectoryDb &m_original;
    const core::TrajectoryDb &m_simplified;

    std::size_t m_knn_neighbours = 0;

    double m_knn_score = std::numeric_limits<double>::quiet_NaN();
    double m_range_score = std::numeric_limits<double>::quiet_NaN();
    double m_where_at_score = std::numeric_limits<double>::quiet_NaN();
};

} // namespace evaluator
