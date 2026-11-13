#include <gtest/gtest.h>

#include <algorithm>
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
TEST(Rdp, SequencePrefixesMatchEachBudgetIncludingTies) {
    for (const auto &ys : {std::vector<double>{0, 5, 0, 5, 0, 5, 0}, std::vector<double>(7, 0.0)}) {
        const Trajectory traj({0, 1, 2, 3, 4, 5, 6}, ys);
        for (std::size_t cap = 2; cap <= traj.x.size(); ++cap) {
            const auto sequence = selection::rdp::sequence(traj.view(), cap);
            ASSERT_EQ(sequence.size(), cap);
            for (std::size_t b = 2; b <= cap; ++b) {
                Indices kept(sequence.begin(), sequence.begin() + b);
                std::ranges::sort(kept);
                EXPECT_EQ(kept, selection::rdp::simplify(traj.view(), b));
            }
        }
    }
    EXPECT_TRUE(selection::rdp::sequence(Trajectory({}, {}).view(), 5).empty());
    EXPECT_EQ(selection::rdp::sequence(Trajectory({0}, {0}).view(), 5), (Indices{0}));
}
