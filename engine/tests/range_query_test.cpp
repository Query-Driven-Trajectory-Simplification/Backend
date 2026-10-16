#include <gtest/gtest.h>

#include "backend/engine/query/range_query.hpp"

#include <algorithm>
#include <vector>

using query::Cell;

namespace {

// Cell A and B are next to each other in the first week.
constexpr double kLatA = 39.41;
constexpr double kLatB = 39.45;
constexpr double kLon = 115.71;
constexpr std::int64_t kT = query::kOriginT + 1;

} // namespace

TEST(CellOf, BeijingPoint) {
    // (39.984702 - 39.4) / 0.02 = 29.2, (116.318417 - 115.7001) / 0.02 = 30.9,
    // (1224730384 - 1176341492) / 604800 = 80.01
    const auto c = query::cell_of(39.984702, 116.318417, 1224730384);
    ASSERT_TRUE(c);
    EXPECT_EQ(*c, (Cell{29, 30, 80}));
}

TEST(CellOf, OutsideBeijing) {
    EXPECT_FALSE(query::cell_of(39.3, 116.3, kT));
    EXPECT_FALSE(query::cell_of(40.0, 118.0, kT));
}

TEST(CellOf, BeforeOriginIsNegativeWeek) {
    EXPECT_EQ(query::cell_of(kLatA, kLon, query::kOriginT - 1)->t, -1);
    EXPECT_EQ(query::cell_of(kLatA, kLon, query::kOriginT)->t, 0);
}

TEST(HeatMap, SkipsEndpointsAndCountsTrajectoriesOnce) {
    // Trajectory 0 starts and ends in A and visits B twice, trajectory 1 stays in B.
    const core::TrajectoryDb db({0, 4, 7}, {0, 1}, {kT, kT + 1, kT + 2, kT + 3, kT, kT + 1, kT + 2},
                                {kLatA, kLatB, kLatB, kLatA, kLatB, kLatB, kLatB},
                                {kLon, kLon, kLon, kLon, kLon, kLon, kLon});
    const auto heat = query::heat_map(db);
    EXPECT_FALSE(heat.contains(*query::cell_of(kLatA, kLon, kT)));
    EXPECT_EQ(heat.at(*query::cell_of(kLatB, kLon, kT)), 2u);
}

TEST(SampleQueries, SameSeedSameQueries) {
    const query::HeatMap heat = {{{1, 1, 1}, 5}, {{2, 2, 2}, 3}, {{3, 3, 3}, 9}};
    const auto first = query::sample_queries(heat, 42);
    const auto second = query::sample_queries(heat, 42);
    ASSERT_EQ(first.size(), query::kQueriesPerSeed);
    for (std::size_t i = 0; i < first.size(); ++i) {
        EXPECT_EQ(first[i].cell, second[i].cell);
    }
}

TEST(SampleQueries, SkipsCellsBelowMinWeight) {
    const Cell low{1, 1, 1}, high{2, 2, 2};
    const query::HeatMap heat = {{low, 1}, {high, 2}};
    for (const auto &q : query::sample_queries(heat, 7, 1000)) {
        EXPECT_EQ(q.cell, high);
    }
}

TEST(SampleQueries, ProportionalToWeight) {
    const Cell a{1, 1, 1}, b{2, 2, 2};
    const query::HeatMap heat = {{a, 2}, {b, 6}};
    const auto queries = query::sample_queries(heat, 1, 100000);
    const auto hits_b = std::count_if(queries.begin(), queries.end(),
                                      [&](const query::RangeQuery &q) { return q.cell == b; });
    EXPECT_NEAR(static_cast<double>(hits_b) / 100000.0, 0.75, 0.01);
}

TEST(SampleQueries, EmptyWhenNoCellQualifies) {
    const query::HeatMap heat = {{{1, 1, 1}, 1}};
    EXPECT_TRUE(query::sample_queries(heat, 1).empty());
}

TEST(SampleQueries, BoundsMatchCell) {
    const Cell c{29, 30, 80};
    const query::HeatMap heat = {{c, 4}};
    const auto q = query::sample_queries(heat, 3, 1).front();
    EXPECT_EQ(q.weight, 4u);
    EXPECT_DOUBLE_EQ(q.bounds.min_lat, query::bounds_of(c).min_lat);
    EXPECT_EQ(q.bounds.t_start, query::kOriginT + 80 * query::kCellSeconds);
}
