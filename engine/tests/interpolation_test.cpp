#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "backend/engine/core/interpolation.hpp"
#include "backend/engine/core/trajectory_db.hpp"

namespace {

struct Trajectory {
    std::vector<std::int64_t> t;
    std::vector<double> x;
    std::vector<double> y;

    core::TrajectoryView view() const { return {0, t, {}, {}, x, y}; }
};

void expect_position(const core::LocalPoint &p, double x, double y) {
    EXPECT_DOUBLE_EQ(p.x, x);
    EXPECT_DOUBLE_EQ(p.y, y);
}

} // namespace

TEST(PositionAt, ReturnsPointAtItsTimestamp) {
    const Trajectory traj{{0, 10, 20}, {0, 100, 100}, {0, 0, 0}};
    expect_position(core::position_at(traj.view(), 10), 100, 0);
}

TEST(PositionAt, ReturnsEndpointsAtSpanBounds) {
    const Trajectory traj{{0, 10, 20}, {0, 100, 300}, {5, 0, 7}};
    expect_position(core::position_at(traj.view(), 0), 0, 5);
    expect_position(core::position_at(traj.view(), 20), 300, 7);
}

TEST(PositionAt, InterpolatesLinearlyInTime) {
    const Trajectory traj{{0, 10, 20}, {0, 100, 100}, {0, 0, 0}};
    expect_position(core::position_at(traj.view(), 5), 50, 0);
    expect_position(core::position_at(traj.view(), 2.5), 25, 0);
}

TEST(PositionAt, StaysPutWhileStopped) {
    const Trajectory traj{{0, 10, 20}, {0, 100, 100}, {0, 0, 0}};
    expect_position(core::position_at(traj.view(), 15), 100, 0);
}

TEST(PositionAt, InterpolatesBothCoordinatesOnUnevenSampling) {
    // 1 s then 9 s between points: t = 4 is a third of the way through the second segment
    const Trajectory traj{{0, 1, 10}, {0, 10, 40}, {0, -10, 80}};
    expect_position(core::position_at(traj.view(), 4), 20, 20);
}

TEST(PositionAt, SinglePointTrajectory) {
    const Trajectory traj{{7}, {3}, {4}};
    expect_position(core::position_at(traj.view(), 7), 3, 4);
}