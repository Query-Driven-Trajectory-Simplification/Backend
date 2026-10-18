#include <gtest/gtest.h>

#include "backend/engine/query/box_query.hpp"

#include <set>

namespace {

constexpr std::int64_t kT = 1'200'000'000;
constexpr std::int64_t kDay = 24 * 60 * 60;
constexpr double kLat = 39.9;
constexpr double kLon = 116.3;
constexpr double kStep = 0.001;

core::TrajectoryDb make_db() {
    return core::TrajectoryDb(
        {0, 3, 6, 9, 12}, {10, 11, 12, 13},
        {kT, kT + 60, kT + 120, kT + 30, kT + 90, kT + 150, kT + 10 * kDay, kT + 10 * kDay + 60,
         kT + 10 * kDay + 120, kT, kT + 60, kT + 120},
        {kLat, kLat + kStep, kLat + 2 * kStep, kLat, kLat + kStep, kLat + 2 * kStep, kLat,
         kLat + kStep, kLat + 2 * kStep, kLat + 1, kLat + 1, kLat + 1},
        {kLon, kLon, kLon, kLon, kLon, kLon, kLon, kLon, kLon, kLon, kLon, kLon});
}

query::BoxQuery box_around(double lat, double lon, std::int64_t t, double half_deg,
                           std::int64_t half_window) {
    return {.trajectory_id = 0,
            .lat = lat,
            .lon = lon,
            .t = t,
            .min_lat = lat - half_deg,
            .max_lat = lat + half_deg,
            .min_lon = lon - half_deg,
            .max_lon = lon + half_deg,
            .t_start = t - half_window,
            .t_end = t + half_window,
            .results = 0};
}

} // namespace

TEST(CountResults, CountsDistinctTrajectoriesInBoxAndWindow) {
    const auto db = make_db();
    const query::BoxQuerySampler sampler(db);
    EXPECT_EQ(sampler.count_results(box_around(kLat, kLon, kT, 0.01, kDay)), 2u);
}

TEST(CountResults, WiderWindowAddsTheLaterTrajectory) {
    const auto db = make_db();
    const query::BoxQuerySampler sampler(db);
    EXPECT_EQ(sampler.count_results(box_around(kLat, kLon, kT, 0.01, 11 * kDay)), 3u);
}

TEST(CountResults, SmallBoxOnlyCatchesItsOwnPoint) {
    const auto db = make_db();
    const query::BoxQuerySampler sampler(db);
    EXPECT_EQ(sampler.count_results(box_around(kLat, kLon, kT, 0.0001, 10)), 1u);
}

TEST(CountResults, ZeroOutsideTheDataInTime) {
    const auto db = make_db();
    const query::BoxQuerySampler sampler(db);
    EXPECT_EQ(sampler.count_results(box_around(kLat, kLon, kT - 100 * kDay, 0.01, kDay)), 0u);
    EXPECT_EQ(sampler.count_results(box_around(kLat, kLon, kT + 100 * kDay, 0.01, kDay)), 0u);
}

TEST(SampleBoxes, SameSeedSameBoxes) {
    const auto db = make_db();
    const query::BoxQuerySampler sampler(db);
    const query::BoxQueryParams params{.min_results = 1, .full_accept_at = 1, .count = 50};
    const auto first = sampler.sample(42, params);
    const auto second = sampler.sample(42, params);
    ASSERT_EQ(first.size(), 50u);
    ASSERT_EQ(second.size(), 50u);
    for (std::size_t i = 0; i < first.size(); ++i) {
        EXPECT_EQ(first[i].trajectory_id, second[i].trajectory_id);
        EXPECT_EQ(first[i].t, second[i].t);
    }
}

TEST(SampleBoxes, ResultsMatchCountAndRule) {
    const auto db = make_db();
    const query::BoxQuerySampler sampler(db);
    const query::BoxQueryParams params{.box_km = 2,
                                       .window_seconds = 2 * kDay,
                                       .min_results = 2,
                                       .full_accept_at = 2,
                                       .count = 40};
    const auto boxes = sampler.sample(7, params);
    ASSERT_EQ(boxes.size(), 40u);
    for (const auto &q : boxes) {
        EXPECT_EQ(q.results, sampler.count_results(q));
        EXPECT_GE(q.results, 2u);
        EXPECT_EQ(q.t_end - q.t_start, params.window_seconds);
        EXPECT_NEAR(q.max_lat - q.lat, q.lat - q.min_lat, 1e-9);
    }
}

TEST(SampleBoxes, OnlyKeepsBoxesWithEnoughResults) {
    const auto db = make_db();
    const query::BoxQuerySampler sampler(db);
    const query::BoxQueryParams params{.box_km = 2,
                                       .window_seconds = 2 * kDay,
                                       .min_results = 2,
                                       .full_accept_at = 2,
                                       .count = 100};
    std::set<std::int32_t> centres;
    for (const auto &q : sampler.sample(3, params)) {
        centres.insert(q.trajectory_id);
    }
    EXPECT_EQ(centres, (std::set<std::int32_t>{10, 11}));
}

TEST(SampleBoxes, GivesUpWhenNoBoxQualifies) {
    const auto db = make_db();
    const query::BoxQuerySampler sampler(db);
    const query::BoxQueryParams params{.min_results = 50, .full_accept_at = 50, .count = 3};
    EXPECT_TRUE(sampler.sample(1, params).empty());
}
