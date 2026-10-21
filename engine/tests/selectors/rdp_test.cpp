#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/selection/rdp.hpp"

namespace {

struct Trajectory {
    std::vector<std::int64_t> t;
    std::vector<double> x;
    std::vector<double> y;

    Trajectory(std::vector<double> xs, std::vector<double> ys)
        : x(std::move(xs)), y(std::move(ys)) {
        for (std::size_t i = 0; i < x.size(); ++i)
            t.push_back(static_cast<std::int64_t>(i));
    }

    core::TrajectoryView view() const { return {0, t, {}, {}, x, y}; }
};

using Indices = std::vector<std::size_t>;

} // namespace

TEST(Rdp, KeepsOnlyEndpointsWithBudgetOfTwo) {
    const Trajectory traj({0, 1, 2, 3, 4}, {0, 5, 0, 5, 0});
    EXPECT_EQ(selection::rdp::simplify(traj.view(), 2), (Indices{0, 4}));
}

TEST(Rdp, KeepsEverythingWhenBudgetCoversTrajectory) {
    const Trajectory traj({0, 1, 2}, {0, 1, 0});
    EXPECT_EQ(selection::rdp::simplify(traj.view(), 3), (Indices{0, 1, 2}));
    EXPECT_EQ(selection::rdp::simplify(traj.view(), 10), (Indices{0, 1, 2}));
}

TEST(Rdp, AddsLargestErrorFirst) {
    const Trajectory traj({0, 1, 2, 3, 4, 5, 6}, {0, 0, 10, 0, 3, 0, 0});
    EXPECT_EQ(selection::rdp::simplify(traj.view(), 3), (Indices{0, 2, 6}));
    EXPECT_EQ(selection::rdp::simplify(traj.view(), 4), (Indices{0, 2, 3, 6}));
}

TEST(Rdp, HandlesRoundTrip) {
    const Trajectory traj({0, 10, 10, 0, 0}, {0, 0, 10, 10, 0});
    EXPECT_EQ(selection::rdp::simplify(traj.view(), 3), (Indices{0, 2, 4}));
}