#pragma once

#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/evaluator/evaluator.hpp"
#include "backend/engine/pipeline/config.hpp"
#include "backend/engine/query/join.hpp"
#include "backend/engine/query/knn.hpp"
#include "backend/engine/query/range_query.hpp"
#include "backend/engine/query/route_clustering.hpp"
#include "backend/engine/query/where_at_query.hpp"

#include <optional>

namespace pipeline {

// The held-out workloads, generated in declaration order. Not copyable: view() points into it
struct TestWorkloads {
    query::KnnWorkload knn;
    query::WhereAtWorkload where_at;
    query::RangeWorkload range;
    query::JoinWorkload join;
    std::vector<query::JoinWorkload> join_grid; // the join at config.join_score_eps_m
    std::optional<query::JoinWorkload> self_join;
    std::optional<query::RouteClusteringWorkload> clustering;
    std::vector<query::RouteClusteringWorkload> clustering_grid; // at config.join_score_eps_m
    // For J4-self. The self join at config.join_score_eps_m, empty when join_grid already is
    std::vector<query::JoinWorkload> self_join_grid;
    std::optional<query::KnnWorkload> self_knn; // supplied, as its reference is slow to build

    TestWorkloads(const core::TrajectoryDb &database, const core::TrajectoryDb &queries,
                  const Config &config, std::optional<query::KnnWorkload> self_knn = std::nullopt);
    TestWorkloads(const TestWorkloads &) = delete;
    TestWorkloads &operator=(const TestWorkloads &) = delete;

    evaluator::Workloads view() const;
};

} // namespace pipeline
