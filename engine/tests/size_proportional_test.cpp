#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "backend/engine/allocation/constants.hpp"
#include "backend/engine/allocation/size_proportional.hpp"
#include "backend/engine/core/trajectory_db.hpp"

namespace {

// Builds a TrajectoryDb with one trajectory per entry in sizes. Point values don't matter here,
// only the counts, but timestamps must be strictly increasing within each trajectory.
core::TrajectoryDb MakeDb(const std::vector<std::size_t> &sizes) {
    std::vector<std::uint64_t> offsets{0};
    std::vector<std::int32_t> ids;
    std::vector<std::int64_t> t;
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        for (std::size_t k = 0; k < sizes[i]; ++k) {
            t.push_back(static_cast<std::int64_t>(k));
        }
        offsets.push_back(t.size());
        ids.push_back(static_cast<std::int32_t>(i));
    }
    std::vector<double> lat(t.size());
    std::vector<double> lon(t.size());
    return core::TrajectoryDb(offsets, ids, t, lat, lon);
}

std::size_t Sum(const std::vector<std::size_t> &v) {
    return std::accumulate(v.begin(), v.end(), std::size_t{0});
}

} // namespace

TEST(ProportionalBudget, BudgetCoversAllPointsKeepsEverything) {
    const std::vector<std::size_t> sizes = {5, 8, 3};
    EXPECT_EQ(allocation::proportional_budget(MakeDb(sizes), 16), sizes);
    EXPECT_EQ(allocation::proportional_budget(MakeDb(sizes), 100), sizes);
}

TEST(ProportionalBudget, ExactlyProportionalWhenSharesDivideEvenly) {
    const std::vector<std::size_t> expected = {5, 10, 15};
    EXPECT_EQ(allocation::proportional_budget(MakeDb({10, 20, 30}), 30), expected);
}

TEST(ProportionalBudget, LeftoverGoesToSmallestBudgetToSizeRatio) {
    // Rounded-down shares are 3 (3/30 = 0.10) and 6 (6/50 = 0.12), so the leftover point goes to
    // trajectory 0.
    const std::vector<std::size_t> expected = {4, 6};
    EXPECT_EQ(allocation::proportional_budget(MakeDb({30, 50}), 10), expected);
}

TEST(ProportionalBudget, LeftoverPrefersSmallTrajectoryWithLowerRatio) {
    // Rounded-down shares are 20 (20/205 ~ 0.098) and 2 (2/25 = 0.08), so the small trajectory gets
    // the point.
    const std::vector<std::size_t> expected = {20, 3};
    EXPECT_EQ(allocation::proportional_budget(MakeDb({205, 25}), 23), expected);
}

TEST(ProportionalBudget, LeftoversNeverExceedTrajectorySize) {
    const auto budgets = allocation::proportional_budget(MakeDb({100, 100, 100, 100, 100}), 499);
    EXPECT_EQ(Sum(budgets), 499u);
    for (std::size_t b : budgets) {
        EXPECT_LE(b, 100u);
    }
}

TEST(ProportionalBudget, OvershootFromMinimumTakenFromLargestRatio) {
    // Each small trajectory is clamped up to 2, so points are taken back from the large one.
    const std::vector<std::size_t> expected = {2, 2, 2, 2};
    EXPECT_EQ(allocation::proportional_budget(MakeDb({1000, 2, 2, 2}), 8), expected);
}

TEST(ProportionalBudget, OvershootNeverGoesBelowMinimum) {
    const std::vector<std::size_t> expected = {3, 2, 2, 2};
    EXPECT_EQ(allocation::proportional_budget(MakeDb({1000, 3, 3, 3}), 9), expected);
}

TEST(ProportionalBudget, ThrowsWhenBudgetBelowMinimumPerTrajectory) {
    EXPECT_THROW(allocation::proportional_budget(MakeDb({10, 10, 10}), 5), std::invalid_argument);
}

TEST(ProportionalBudget, ThrowsWhenTrajectoryHasTooFewPoints) {
    EXPECT_THROW(allocation::proportional_budget(MakeDb({10, 1, 10}), 10), std::invalid_argument);
}

TEST(ProportionalBudget, RandomInputsSatisfyInvariants) {
    std::mt19937 rng(42);
    for (int iteration = 0; iteration < 2000; ++iteration) {
        const std::size_t num_trajectories = 1 + rng() % 20;
        std::vector<std::size_t> sizes(num_trajectories);
        std::size_t num_points = 0;
        for (std::size_t &size : sizes) {
            size = allocation::kMinPointsPerTrajectory + rng() % 500;
            num_points += size;
        }
        const std::size_t min_budget = allocation::kMinPointsPerTrajectory * num_trajectories;
        const std::size_t budget = min_budget + rng() % (num_points - min_budget + 1);

        const auto budgets = allocation::proportional_budget(MakeDb(sizes), budget);

        ASSERT_EQ(budgets.size(), num_trajectories);
        EXPECT_EQ(Sum(budgets), budget);
        for (std::size_t i = 0; i < num_trajectories; ++i) {
            EXPECT_GE(budgets[i], allocation::kMinPointsPerTrajectory);
            EXPECT_LE(budgets[i], sizes[i]);
        }
    }
}
