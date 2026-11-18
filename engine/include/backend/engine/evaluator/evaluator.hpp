#pragma once

#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/evaluator/join.hpp"
#include "backend/engine/evaluator/knn.hpp"
#include "backend/engine/evaluator/route_clustering.hpp"
#include "backend/engine/evaluator/where_at.hpp"
#include "backend/engine/query/join.hpp"
#include "backend/engine/query/knn.hpp"
#include "backend/engine/query/range_query.hpp"
#include "backend/engine/query/route_clustering.hpp"

#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace evaluator {

// A missing workload leaves its scores NaN. Workloads must outlive the evaluator.
struct Workloads {
    const query::KnnWorkload *knn = nullptr; // a pointer, as it also carries k and the query db
    std::span<const query::WhereAtQuery> where_at = {};
    std::span<const query::RangeQuery> range = {};
    const query::JoinWorkload *join = nullptr; // a pointer, as it also carries eps and the queries
    const query::RouteClusteringWorkload *clustering = nullptr;
    // Diagnostic; may point at join itself
    const query::JoinWorkload *self_join = nullptr;
    // Extra thresholds, averaged into the reported scores
    std::span<const query::JoinWorkload> join_grid = {};
    std::span<const query::RouteClusteringWorkload> clustering_grid = {};
};

// Kendall's tau is in [-1, 1] and the others in [0, 1], or NaN if it could not be computed
struct Scoretable {
    KnnScores knn;
    double range_recall;
    WhereAtScores where_at;
    JoinScores join;
    double range_f1 = std::numeric_limits<double>::quiet_NaN();
    RouteClusteringScores clustering;
    JoinScores self_join;
    std::vector<JoinScores> join_grid;                  // in Workloads::join_grid order
    std::vector<RouteClusteringScores> clustering_grid; // in Workloads::clustering_grid order
};

struct EvaluationResult {
    Scoretable scores;
    std::vector<KnnAnswer> knn;
    std::vector<std::vector<std::int32_t>> range;
    std::vector<JoinPair> join;
    std::vector<std::vector<std::int32_t>> clusters;
};

class Evaluator {
  public:
    Evaluator(const core::TrajectoryDb &original, const core::TrajectoryDb &simplified,
              Workloads workloads = {}, WhereAtScoringConfig where_at_config = {});

    EvaluationResult evaluate() const;

    // Each runs only its own evaluation
    KnnScores get_knn_scores() const;
    double get_range_recall() const;
    double get_range_f1() const;
    WhereAtScores get_where_at_scores() const;
    JoinScores get_join_scores() const;
    JoinScores get_self_join_scores() const;
    RouteClusteringScores get_clustering_scores() const;

  private:
    const core::TrajectoryDb &m_simplified;
    Workloads m_workloads;
    WhereAtScoringConfig m_where_at_config;
};

} // namespace evaluator
