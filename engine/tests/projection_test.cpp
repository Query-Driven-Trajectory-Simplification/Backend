#include <gtest/gtest.h>

#include <stdexcept>

#include "backend/engine/core/projection.hpp"
#include "backend/engine/core/trajectory_db.hpp"

TEST(Projection, OriginIsZero) {
    const auto p = core::to_local_metres(core::origin_lat, core::origin_lon);
    EXPECT_DOUBLE_EQ(p.x, 0.0);
    EXPECT_DOUBLE_EQ(p.y, 0.0);
}

TEST(Projection, HundredthOfDegreeInMetres) {
    // 0.01 degree is about 1110 m north and 854.6 m east at Beijing latitude.
    const auto north = core::to_local_metres(core::origin_lat + 0.01, core::origin_lon);
    const auto east = core::to_local_metres(core::origin_lat, core::origin_lon + 0.01);
    EXPECT_NEAR(north.y, 1110.3, 1.0);
    EXPECT_DOUBLE_EQ(north.x, 0.0);
    EXPECT_NEAR(east.x, 854.6, 1.0);
    EXPECT_DOUBLE_EQ(east.y, 0.0);
}

TEST(TrajectoryDb, StoresProjectedPoints) {
    const core::TrajectoryDb db({0, 2}, {7}, {10, 11}, {39.95, 40.0}, {116.35, 116.4});
    const auto trip = db.trajectory(0);
    ASSERT_EQ(trip.x.size(), 2u);
    const auto expected = core::to_local_metres(40.0, 116.4);
    EXPECT_DOUBLE_EQ(trip.x[0], 0.0);
    EXPECT_DOUBLE_EQ(trip.y[0], 0.0);
    EXPECT_DOUBLE_EQ(trip.x[1], expected.x);
    EXPECT_DOUBLE_EQ(trip.y[1], expected.y);
}

TEST(TrajectoryDb, RejectsRepeatedTimestamps) {
    EXPECT_THROW(core::TrajectoryDb({0, 2}, {7}, {10, 10}, {39.95, 40.0}, {116.35, 116.4}),
                 std::invalid_argument);
}