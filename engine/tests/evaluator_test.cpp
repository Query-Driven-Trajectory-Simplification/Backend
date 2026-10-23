#include "backend/engine/core/simplification.hpp"
#include "backend/engine/evaluator/evaluator.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

core::TrajectoryDb original() {
    return {{0, 3}, {10}, {0, 10, 20}, {39.95, 39.95, 39.95}, {116.35, 116.36, 116.36}};
}

core::TrajectoryDb endpoints(const core::TrajectoryDb &db) {
    core::Simplification simplification;
    const std::vector<std::size_t> kept{0, 2};
    simplification.add_trajectory(kept);
    return core::materialize(db, simplification);
}

} // namespace

TEST(Evaluator, CollectsBothWhereAtScoresWithoutUnconfiguredQueries) {
    const auto db = original();
    const auto workload = query::generate_where_at_queries(db);
    evaluator::Evaluator evaluator(db, db, {.where_at = workload.queries});
    const auto scores = evaluator.evaluate();
    EXPECT_DOUBLE_EQ(scores.where_at.within_radius, 1);
    EXPECT_DOUBLE_EQ(scores.where_at.average_hit, 1);
    EXPECT_TRUE(std::isnan(scores.knn_score));
    EXPECT_TRUE(std::isnan(scores.range_score));
}

TEST(Evaluator, DelegatesToWhereAtModule) {
    const auto db = original();
    const auto compressed = endpoints(db);
    const auto workload = query::generate_where_at_queries(db);
    const evaluator::WhereAtScoringConfig config{200, 50, 250};
    const auto expected = evaluator::evaluate_where_at(compressed, workload.queries, config);
    evaluator::Evaluator evaluator(db, compressed, {.where_at = workload.queries}, config);
    EXPECT_DOUBLE_EQ(evaluator.get_where_at_scores().within_radius, expected.within_radius);
    EXPECT_DOUBLE_EQ(evaluator.evaluate().where_at.average_hit,
                     expected.average_hit);
}

TEST(Evaluator, ReusesSameWorkloadAcrossDatabases) {
    const auto db = original();
    const auto compressed = endpoints(db);
    const auto workload = query::generate_where_at_queries(db);
    evaluator::Evaluator baseline(db, db, {.where_at = workload.queries});
    evaluator::Evaluator simplified(db, compressed, {.where_at = workload.queries});
    EXPECT_DOUBLE_EQ(baseline.get_where_at_scores().within_radius, 1);
    EXPECT_LT(simplified.get_where_at_scores().within_radius, 1);
}

TEST(Evaluator, NoWorkloadsReturnsUndefinedScores) {
    const auto db = original();
    evaluator::Evaluator evaluator(db, db);
    const auto scores = evaluator.evaluate();
    EXPECT_TRUE(std::isnan(scores.knn_score));
    EXPECT_TRUE(std::isnan(scores.range_score));
    EXPECT_TRUE(std::isnan(scores.where_at.within_radius));
    EXPECT_TRUE(std::isnan(scores.where_at.average_hit));
}

TEST(Evaluator, RejectsDifferentTrajectoryCountsAndIds) {
    const auto db = original();
    const core::TrajectoryDb empty({0}, {}, {}, {}, {});
    const core::TrajectoryDb wrong_id({0, 2}, {99}, {0, 20}, {39.95, 39.95}, {116.35, 116.36});
    EXPECT_THROW((evaluator::Evaluator(db, empty)), std::invalid_argument);
    EXPECT_THROW((evaluator::Evaluator(db, wrong_id)), std::invalid_argument);
}
