#include "backend/engine/core/projection.hpp"
#include "backend/engine/core/simplification.hpp"
#include "backend/engine/evaluator/range.hpp"
#include "backend/engine/query/range_query.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

namespace {

struct Point {
    std::int64_t t;
    double lat = core::origin_lat;
    double lon = core::origin_lon;
};

core::TrajectoryDb make_db(const std::vector<std::vector<Point>> &trajectories) {
    std::vector<std::uint64_t> offsets{0};
    std::vector<std::int32_t> ids;
    std::vector<std::int64_t> t;
    std::vector<double> lat;
    std::vector<double> lon;
    for (std::size_t i = 0; i < trajectories.size(); ++i) {
        ids.push_back(static_cast<std::int32_t>(10 + i));
        for (const auto &p : trajectories[i]) {
            t.push_back(p.t);
            lat.push_back(p.lat);
            lon.push_back(p.lon);
        }
        offsets.push_back(t.size());
    }
    return {offsets, ids, t, lat, lon};
}

// A box around the projection origin, where x = y = 0
query::RangeBox origin_box(std::int64_t t_min, std::int64_t t_max) {
    return {-100, 100, -100, 100, t_min, t_max};
}

using Indices = std::vector<std::size_t>;

} // namespace

TEST(RangeQueries, BoxEdgesAreIncluded) {
    const auto db = make_db({{{1000}}});
    for (const query::RangeBox box :
         {query::RangeBox{0, 10, -10, 10, 900, 1100}, query::RangeBox{-10, 0, -10, 10, 900, 1100},
          query::RangeBox{-10, 10, 0, 10, 900, 1100}, query::RangeBox{-10, 10, -10, 0, 900, 1100},
          query::RangeBox{-10, 10, -10, 10, 1000, 1100},
          query::RangeBox{-10, 10, -10, 10, 900, 1000}})
        EXPECT_EQ(query::range(db, box), Indices{0});
}

TEST(RangeQueries, OutsideInAnyDimensionIsOut) {
    const auto db = make_db({{{1000}}});
    for (const query::RangeBox box :
         {query::RangeBox{1, 10, -10, 10, 900, 1100}, query::RangeBox{-10, -1, -10, 10, 900, 1100},
          query::RangeBox{-10, 10, 1, 10, 900, 1100}, query::RangeBox{-10, 10, -10, -1, 900, 1100},
          query::RangeBox{-10, 10, -10, 10, 1001, 1100},
          query::RangeBox{-10, 10, -10, 10, 900, 999}})
        EXPECT_TRUE(query::range(db, box).empty());
}

TEST(RangeQueries, ReturnsEachMatchingTrajectoryOnceInOrder) {
    const auto db = make_db({
        {{0}, {10}, {20}},                     // inside at every time
        {{500}},                               // inside space, outside the window
        {{0, 40.0, 117.0}, {15}},              // only the second point is inside
        {{5, 40.0, 117.0}, {15, 40.0, 117.0}}, // outside space
    });
    EXPECT_EQ(query::range(db, origin_box(0, 100)), (Indices{0, 2}));
}

TEST(RangeQueries, TrajectoryFullyInsideMatches) {
    const auto db = make_db({{
        {0, core::origin_lat + 0.0005},
        {10, core::origin_lat, core::origin_lon + 0.0005},
        {20, core::origin_lat - 0.0005, core::origin_lon - 0.0005},
    }});
    EXPECT_EQ(query::range(db, origin_box(0, 100)), Indices{0});
}

TEST(RangeQueries, OverlappingBoundingBoxWithoutPointsInsideIsOut) {
    // Diagonal across the box: its bounding box covers the box, but neither point is inside
    const auto db = make_db({{
        {0, core::origin_lat - 0.002, core::origin_lon - 0.002},
        {10, core::origin_lat + 0.002, core::origin_lon + 0.002},
    }});
    EXPECT_TRUE(query::range(db, origin_box(0, 100)).empty());
}

TEST(RangeQueries, TimeSpanOverlappingWindowWithoutPointsInsideIsOut) {
    const auto db = make_db({{{-50}, {150}}});
    EXPECT_TRUE(query::range(db, origin_box(0, 100)).empty());
}

TEST(RangeQueries, SkipsEmptyTrajectories) {
    const auto db = make_db({{}, {{0}}});
    EXPECT_EQ(query::range(db, origin_box(0, 100)), Indices{1});
}

TEST(RangeQueries, ReferenceIsTheRangeOfTheOriginal) {
    std::vector<std::vector<Point>> trajectories;
    for (std::int64_t i = 0; i < 30; ++i)
        trajectories.push_back({{i * 100, core::origin_lat + i * 0.001}, {i * 100 + 10}});
    const auto db = make_db(trajectories);
    const auto workload = query::generate_range_queries(db, {50, 1000, 3600, 5, 20, 7});
    ASSERT_FALSE(workload.queries.empty());
    for (const auto &q : workload.queries) {
        EXPECT_EQ(q.reference, query::range(db, q.box));
        EXPECT_GE(q.reference.size(), 5);
        EXPECT_DOUBLE_EQ(q.box.x_max - q.box.x_min, 1000);
        EXPECT_DOUBLE_EQ(q.box.y_max - q.box.y_min, 1000);
        EXPECT_EQ(q.box.t_max - q.box.t_min, 3600);
    }
}

TEST(RangeQueries, PreservesOddWindowDurations) {
    const auto db = make_db({{{100}}});
    for (const std::int64_t seconds : {1, 5}) {
        const auto workload = query::generate_range_queries(db, {1, 1000, seconds, 1, 1, 1});
        ASSERT_EQ(workload.queries.size(), 1);
        const auto &box = workload.queries.front().box;
        EXPECT_EQ(box.t_min, 100 - seconds / 2);
        EXPECT_EQ(box.t_max - box.t_min, seconds);
    }
}

TEST(RangeQueries, RetriesRandomlyRejectedEligibleCentres) {
    const auto db = make_db({{{100}}});
    const auto workload = query::generate_range_queries(db, {1, 1000, 3600, 1, 100, 10000});
    ASSERT_EQ(workload.queries.size(), 1);
    EXPECT_EQ(workload.queries.front().reference, Indices{0});
}

TEST(RangeQueries, RejectsBoxesWithTooFewResults) {
    const auto db = make_db({{{0}, {10}}, {{0}, {10}}, {{0}, {10}}});
    EXPECT_TRUE(query::generate_range_queries(db, {10, 1000, 3600, 4, 4, 1}).queries.empty());
}

TEST(RangeQueries, UsesEachPointAsCentreAtMostOnce) {
    std::vector<std::vector<Point>> trajectories(5, {{0}, {10}});
    const auto db = make_db(trajectories);
    const auto workload = query::generate_range_queries(db, {100, 1000, 3600, 5, 5, 1});
    EXPECT_EQ(workload.queries.size(), 10);
}

TEST(RangeQueries, UsesEachPointOnceAcrossManyBatches) {
    std::vector<std::vector<Point>> trajectories;
    for (std::int64_t i = 0; i < 1000; ++i)
        trajectories.push_back({{10 * i}, {10 * i + 1}, {10 * i + 2}});
    const auto db = make_db(trajectories);
    const auto workload = query::generate_range_queries(db, {5000, 1000, 3600, 1, 1, 10});
    ASSERT_EQ(workload.queries.size(), 3000);
    std::set<std::int64_t> centre_times;
    for (const auto &query : workload.queries)
        centre_times.insert(query.box.t_min);
    EXPECT_EQ(centre_times.size(), 3000);
}

TEST(RangeQueries, Reproducible) {
    std::vector<std::vector<Point>> trajectories;
    for (std::int64_t i = 0; i < 30; ++i)
        trajectories.push_back({{i * 100, core::origin_lat + i * 0.001}, {i * 100 + 10}});
    const auto db = make_db(trajectories);
    const auto a = query::generate_range_queries(db, {20, 1000, 3600, 5, 20, 3});
    const auto b = query::generate_range_queries(db, {20, 1000, 3600, 5, 20, 3});
    ASSERT_EQ(a.queries.size(), b.queries.size());
    for (std::size_t i = 0; i < a.queries.size(); ++i) {
        EXPECT_DOUBLE_EQ(a.queries[i].box.x_min, b.queries[i].box.x_min);
        EXPECT_EQ(a.queries[i].box.t_min, b.queries[i].box.t_min);
        EXPECT_EQ(a.queries[i].reference, b.queries[i].reference);
    }
}

TEST(RangeQueries, EmptyDatabaseHasNoQueries) {
    EXPECT_TRUE(query::generate_range_queries(make_db({})).queries.empty());
    EXPECT_TRUE(query::generate_range_queries(make_db({{}, {}})).queries.empty());
}

TEST(RangeQueries, RejectsInvalidConfiguration) {
    const auto db = make_db({{{0}}});
    for (double size : {0., -1., std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::quiet_NaN()})
        EXPECT_THROW(query::generate_range_queries(db, {10, size, 3600, 5, 20, 1}),
                     std::invalid_argument);
    EXPECT_THROW(query::generate_range_queries(db, {10, 1000, -1, 5, 20, 1}),
                 std::invalid_argument);
    EXPECT_THROW(query::generate_range_queries(db, {10, 1000, 3600, 0, 20, 1}),
                 std::invalid_argument);
}

TEST(RangeRecall, SameResultsScoreOne) {
    EXPECT_DOUBLE_EQ(evaluator::range::recall({1, 2, 3}, {1, 2, 3}), 1.0);
}

TEST(RangeRecall, NoSimplifiedResultsScoreZero) {
    EXPECT_DOUBLE_EQ(evaluator::range::recall({1, 2, 3}, {}), 0.0);
}

TEST(RangeRecall, IsTheShareOfOriginalResultsKept) {
    EXPECT_DOUBLE_EQ(evaluator::range::recall({1, 2, 3, 4}, {1, 2, 4}), 0.75);
}

TEST(RangeRecall, IgnoresResultsNotInTheOriginal) {
    EXPECT_DOUBLE_EQ(evaluator::range::recall({1, 2}, {0, 2, 3}), 0.5);
}

TEST(RangeRecall, NanWhenTheOriginalReturnedNothing) {
    EXPECT_TRUE(std::isnan(evaluator::range::recall({}, {})));
    EXPECT_TRUE(std::isnan(evaluator::range::recall({}, {1})));
}

TEST(RangeRecall, DroppedPointsInsideTheBoxLowerRecall) {
    const auto original = make_db({
        {{0, 40.0, 117.0}, {10}, {20, 40.0, 117.0}},
        {{0, 40.0, 117.0}, {10}, {20, 40.0, 117.0}},
    });
    core::Simplification kept;
    const std::vector<std::size_t> endpoints{0, 2};
    const std::vector<std::size_t> all{0, 1, 2};
    kept.add_trajectory(endpoints);
    kept.add_trajectory(all);
    const auto compressed = core::materialize(original, kept);

    const auto box = origin_box(0, 100);
    EXPECT_DOUBLE_EQ(
        evaluator::range::recall(query::range(original, box), query::range(compressed, box)), 0.5);
}
