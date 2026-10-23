#pragma once

#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/evaluator/where_at.hpp"

#include <cstddef>
#include <limits>
#include <span>

namespace evaluator {

struct Workloads {
    std::span<const query::WhereAtQuery> where_at;
};

struct Scoretable {
    double knn_score;
    double range_score;
    WhereAtScores where_at;
};

class Evaluator {
  public:
    Evaluator(const core::TrajectoryDb &original, const core::TrajectoryDb &simplified,
              Workloads workloads = {}, WhereAtScoringConfig where_at_config = {});

    // Runs every evaluation, stores the scores and returns them
    Scoretable evaluate();

    // Each runs only its own evaluation, stores the score and returns it
    double get_knn_score();
    double get_range_score();
    WhereAtScores get_where_at_scores();

    void set_knn_neighbours(std::size_t k);

  private:
    void knn_score(std::size_t k);
    void range_score();

    const core::TrajectoryDb &m_original;
    const core::TrajectoryDb &m_simplified;
    Workloads m_workloads;
    WhereAtScoringConfig m_where_at_config;

    std::size_t m_knn_neighbours = 0;

    double m_knn_score = std::numeric_limits<double>::quiet_NaN();
    double m_range_score = std::numeric_limits<double>::quiet_NaN();
    WhereAtScores m_where_at_scores;
};

} // namespace evaluator
