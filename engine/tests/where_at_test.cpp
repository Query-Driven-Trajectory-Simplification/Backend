#include "backend/engine/core/interpolation.hpp"
#include "backend/engine/core/simplification.hpp"
#include "backend/engine/evaluator/where_at.hpp"
#include "backend/engine/query/where_at_query.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

core::TrajectoryDb stationary(const std::vector<std::vector<std::int64_t>> &times) {
    std::vector<std::uint64_t> offsets{0};
    std::vector<std::int32_t> ids;
    std::vector<std::int64_t> t;
    for (std::size_t i = 0; i < times.size(); ++i) {
        ids.push_back(static_cast<std::int32_t>(10 + i));
        t.insert(t.end(), times[i].begin(), times[i].end());
        offsets.push_back(t.size());
    }
    return {offsets, ids, t, std::vector<double>(t.size(), 39.95),
            std::vector<double>(t.size(), 116.35)};
}

} // namespace

TEST(WhereAtQueries, StratifiesTimeAndWeightsIntervalsByDuration) {
    const auto db = stationary({{0, 1, 6}, {0, 30}});
    const auto workload = query::generate_where_at_queries(db, {1000, 7, 60});
    ASSERT_EQ(workload.queries.size(), 2000);
    EXPECT_EQ(workload.skipped_trajectories, 0);
    std::size_t first_second = 0;
    for (std::size_t q = 0; q < 1000; ++q) {
        const auto &probe = workload.queries[q];
        EXPECT_EQ(probe.trajectory_index, 0);
        EXPECT_EQ(probe.trajectory_id, 10);
        EXPECT_GE(probe.t, 6.0 * q / 1000);
        EXPECT_LE(probe.t, 6.0 * (q + 1) / 1000);
        first_second += probe.t < 1;
    }
    EXPECT_GE(first_second, 166);
    EXPECT_LE(first_second, 167);
    EXPECT_EQ(workload.queries.back().trajectory_index, 1);
}

TEST(WhereAtQueries, RecordingDensityDoesNotChangeSampledTimes) {
    const auto sparse = stationary({{0, 5, 10}});
    const auto dense = stationary({{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}});
    const auto a = query::generate_where_at_queries(sparse);
    const auto b = query::generate_where_at_queries(dense);
    ASSERT_EQ(a.queries.size(), b.queries.size());
    for (std::size_t i = 0; i < a.queries.size(); ++i)
        EXPECT_NEAR(a.queries[i].t, b.queries[i].t, 1e-12);
}

TEST(WhereAtQueries, ExcludesLongOriginalGapsButIncludesBoundaryLength) {
    const auto db = stationary({{0, 60, 121, 181}});
    const auto workload = query::generate_where_at_queries(db, {1000, 3, 60});
    ASSERT_EQ(workload.queries.size(), 1000);
    std::size_t early = 0;
    for (const auto &q : workload.queries) {
        EXPECT_TRUE(q.t <= 60 || q.t >= 121);
        early += q.t <= 60;
    }
    EXPECT_EQ(early, 500);
}

TEST(WhereAtQueries, SkipsTripsWithoutEligibleTime) {
    const auto db = stationary({{}, {5}, {0, 100}, {0, 10}});
    const auto workload = query::generate_where_at_queries(db, {16, 1, 60});
    EXPECT_EQ(workload.skipped_trajectories, 3);
    ASSERT_EQ(workload.queries.size(), 16);
    EXPECT_EQ(workload.queries.front().trajectory_index, 3);
}

TEST(WhereAtQueries, ReproducibleAndCanGenerateIndependentWorkloads) {
    const auto db = stationary({{0, 10}});
    const auto a = query::generate_where_at_queries(db, {16, 1, 60});
    const auto b = query::generate_where_at_queries(db, {16, 1, 60});
    const auto c = query::generate_where_at_queries(db, {16, 2, 60});
    bool differs = false;
    for (std::size_t i = 0; i < a.queries.size(); ++i) {
        EXPECT_DOUBLE_EQ(a.queries[i].t, b.queries[i].t);
        differs |= a.queries[i].t != c.queries[i].t;
    }
    EXPECT_TRUE(differs);
}

TEST(WhereAtQueries, RejectsInvalidConfiguration) {
    const auto db = stationary({{0, 10}});
    EXPECT_THROW(query::generate_where_at_queries(db, {0, 1, 60}), std::invalid_argument);
    for (double gap : {0., -1., std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()})
        EXPECT_THROW(query::generate_where_at_queries(db, {16, 1, gap}), std::invalid_argument);
}

TEST(WhereAtEvaluation, OriginalDatabaseScoresPerfectly) {
    const core::TrajectoryDb db({0, 3}, {10}, {0, 10, 20}, {39.95, 39.96, 39.955},
                                {116.35, 116.36, 116.38});
    const auto workload = query::generate_where_at_queries(db);
    const auto result = evaluator::evaluate_where_at(db, workload.queries);
    EXPECT_DOUBLE_EQ(result.within_100m, 1);
    EXPECT_DOUBLE_EQ(result.average_hit_25_500m, 1);
}

TEST(WhereAtEvaluation, ComputesBothScoresAtToleranceBoundaries) {
    const auto db = stationary({{0, 10}});
    std::vector<query::WhereAtQuery> queries;
    for (double error : {0., 25., 100., 500., 600.})
        queries.push_back({0, 10, 5, {error, 0}});
    const auto result = evaluator::evaluate_where_at(db, queries);
    EXPECT_DOUBLE_EQ(result.within_100m, 3. / 5);
    EXPECT_NEAR(result.average_hit_25_500m, (2 + 400. / 475) / 5, 1e-14);
}

TEST(WhereAtEvaluation, WeightsTripsEquallyWithUnequalQueryCounts) {
    const auto db = stationary({{0, 10}, {0, 10}});
    const std::vector<query::WhereAtQuery> queries{
        {0, 10, 1, {0, 0}}, {1, 11, 5, {200, 0}}, {0, 10, 2, {0, 0}}, {0, 10, 3, {0, 0}}};
    const auto result = evaluator::evaluate_where_at(db, queries);
    EXPECT_DOUBLE_EQ(result.within_100m, .5);
    EXPECT_NEAR(result.average_hit_25_500m, (1 + 300. / 475) / 2, 1e-14);
}

TEST(WhereAtEvaluation, InterpolatesCompressedTripWithoutFilteringItsLongGaps) {
    const core::TrajectoryDb original({0, 3}, {10}, {0, 50, 100}, {39.95, 39.95, 39.95},
                                      {116.35, 116.36, 116.36});
    core::Simplification kept;
    const std::vector<std::size_t> endpoints{0, 2};
    kept.add_trajectory(endpoints);
    const auto compressed = core::materialize(original, kept);
    const auto truth = core::position_at(original.trajectory(0), 50);
    const std::vector<query::WhereAtQuery> queries{{0, 10, 50, truth}};
    const auto result = evaluator::evaluate_where_at(compressed, queries);
    const auto start = core::position_at(original.trajectory(0), 0);
    const double expected_error = core::distance(start, truth) / 2;
    EXPECT_NEAR(result.average_hit_25_500m, (500 - expected_error) / 475, 1e-12);
    EXPECT_DOUBLE_EQ(result.within_100m, 0);
    const auto workload = query::generate_where_at_queries(original);
    const auto scores = evaluator::evaluate_where_at(compressed, workload.queries);
    EXPECT_TRUE(std::isfinite(scores.within_100m));
    EXPECT_TRUE(std::isfinite(scores.average_hit_25_500m));
}

TEST(WhereAtEvaluation, RejectsMissingMismatchedAndOutOfSpanQueries) {
    const auto db = stationary({{0, 10}});
    for (const auto &q : std::vector<query::WhereAtQuery>{
             {1, 10, 5, {0, 0}},
             {0, 999, 5, {0, 0}},
             {0, 10, -1, {0, 0}},
             {0, 10, 11, {0, 0}},
             {0, 10, std::numeric_limits<double>::quiet_NaN(), {0, 0}},
             {0, 10, 5, {std::numeric_limits<double>::infinity(), 0}}}) {
        const std::vector<query::WhereAtQuery> queries{q};
        EXPECT_THROW(evaluator::evaluate_where_at(db, queries), std::invalid_argument);
    }
    const auto empty_trip = stationary({{}});
    const std::vector<query::WhereAtQuery> queries{{0, 10, 0, {0, 0}}};
    EXPECT_THROW(evaluator::evaluate_where_at(empty_trip, queries), std::invalid_argument);
}

TEST(WhereAtEvaluation, EmptyWorkloadHasNoScore) {
    const auto db = stationary({});
    EXPECT_TRUE(query::generate_where_at_queries(db).queries.empty());
    const auto result = evaluator::evaluate_where_at(db, {});
    EXPECT_TRUE(std::isnan(result.within_100m));
    EXPECT_TRUE(std::isnan(result.average_hit_25_500m));
}
