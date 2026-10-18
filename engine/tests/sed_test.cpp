#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/selection/sed.hpp"

namespace {

struct Trajectory {
    std::vector<std::int64_t> t;
    std::vector<double> x;
    std::vector<double> y;

    Trajectory(std::vector<double> xs, std::vector<double> ys) : x(std::move(xs)), y(std::move(ys)) {
        for (std::size_t i = 0; i < x.size(); ++i)
            t.push_back(static_cast<std::int64_t>(i));
    }

    core::TrajectoryView view() const { return {0, t, {}, {}, x, y}; }
};

using Indices = std::vector<std::size_t>;

// Max SED of every original point against the simplified trajectory
double max_sed(const core::TrajectoryView &trajectory, const Indices &kept) {
    double error = 0.0;
    for (std::size_t s = 0; s + 1 < kept.size(); ++s) {
        const std::size_t first = kept[s];
        const std::size_t last = kept[s + 1];
        for (std::size_t k = first + 1; k < last; ++k)
            error = std::max(error,
                             selection::sed::synchronous_distance(trajectory, first, last, k));
    }
    return error;
}

}

TEST(Sed, SynchronousDistanceUsesTimeNotShape) {
    // On the line but ahead of schedule: perpendicular distance is 0, SED is not
    const Trajectory traj({0, 8, 10}, {0, 0, 0});
    EXPECT_DOUBLE_EQ(selection::sed::synchronous_distance(traj.view(), 0, 2, 1), 3.0);
}

TEST(Sed, KeepsEverythingWhenBudgetCoversTrajectory) {
    const Trajectory traj({0, 1, 2}, {0, 1, 0});
    EXPECT_EQ(selection::sed::simplify(traj.view(), 3), (Indices{0, 1, 2}));
    EXPECT_EQ(selection::sed::simplify(traj.view(), 10), (Indices{0, 1, 2}));
}

TEST(Sed, KeepsEndpointsWithBudgetBelowTwo) {
    const Trajectory traj({0, 1, 2, 3}, {0, 5, 0, 5});
    EXPECT_EQ(selection::sed::simplify(traj.view(), 0), (Indices{0, 3}));
}

TEST(Sed, StraightConstantSpeedNeedsOnlyEndpoints) {
    const Trajectory traj({0, 1, 2, 3, 4}, {0, 2, 4, 6, 8});
    EXPECT_EQ(selection::sed::simplify(traj.view(), 2), (Indices{0, 4}));
}

TEST(Sed, KeepsSpeedChangeOnStraightLine) {
    // Slow for 3 steps then a jump; RDP sees no error here
    const Trajectory traj({0, 1, 2, 3, 10, 11}, {0, 0, 0, 0, 0, 0});
    EXPECT_EQ(selection::sed::simplify(traj.view(), 3), (Indices{0, 3, 5}));
}

TEST(Sed, ReturnsBudgetAndNeverBeatsBruteForce) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> coord(-100.0, 100.0);
    const std::size_t n = 10;

    for (int trial = 0; trial < 50; ++trial) {
        std::vector<double> xs(n), ys(n);
        for (std::size_t i = 0; i < n; ++i) {
            xs[i] = coord(rng);
            ys[i] = coord(rng);
        }
        const Trajectory traj(xs, ys);

        for (std::size_t budget = 2; budget < n; ++budget) {
            const Indices kept = selection::sed::simplify(traj.view(), budget);
            ASSERT_EQ(kept.size(), budget);
            EXPECT_EQ(kept.front(), 0u);
            EXPECT_EQ(kept.back(), n - 1);

            // Best over every interior subset of the right size
            double best = std::numeric_limits<double>::infinity();
            for (std::uint32_t mask = 0; mask < (1u << (n - 2)); ++mask) {
                if (static_cast<std::size_t>(__builtin_popcount(mask)) != budget - 2)
                    continue;
                Indices subset{0};
                for (std::size_t i = 0; i < n - 2; ++i) {
                    if (mask & (1u << i))
                        subset.push_back(i + 1);
                }
                subset.push_back(n - 1);
                best = std::min(best, max_sed(traj.view(), subset));
            }
            EXPECT_GE(max_sed(traj.view(), kept) + 1e-9, best);
        }
    }
}
