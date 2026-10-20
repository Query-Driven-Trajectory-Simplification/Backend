#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/selection/selector.hpp"

namespace {

// Builds a TrajectoryDb with trajectories of the given sizes, one point per second and random
// coordinates around Beijing
core::TrajectoryDb make_random_db(const std::vector<std::size_t> &sizes) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> offset(-0.01, 0.01);
    std::vector<std::uint64_t> offsets{0};
    std::vector<std::int32_t> ids;
    std::vector<std::int64_t> t;
    std::vector<double> lat;
    std::vector<double> lon;
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        ids.push_back(static_cast<std::int32_t>(i));
        for (std::size_t k = 0; k < sizes[i]; ++k) {
            t.push_back(static_cast<std::int64_t>(k));
            lat.push_back(39.95 + offset(rng));
            lon.push_back(116.35 + offset(rng));
        }
        offsets.push_back(t.size());
    }
    return {std::move(offsets), std::move(ids), std::move(t), std::move(lat), std::move(lon)};
}

constexpr selection::AlgorithmType kAlgorithms[] = {selection::AlgorithmType::RDP,
                                                    selection::AlgorithmType::SED};

} // namespace

TEST(SimplifyDatabase, KeepsBudgetAndEndpointsOfEveryTrajectory) {
    const auto db = make_random_db({2, 5, 20, 50, 8});
    const std::vector<std::size_t> budgets{2, 0, 7, 50, 100};

    for (const auto algorithm : kAlgorithms) {
        const auto simplification = selection::simplify_database(db, algorithm, budgets);

        ASSERT_EQ(simplification.size(), db.size());
        for (std::size_t i = 0; i < db.size(); ++i) {
            const auto kept = simplification.trajectory(i);
            const std::size_t n = db.trajectory(i).size();
            EXPECT_EQ(kept.size(), std::clamp<std::size_t>(budgets[i], 2, n));
            EXPECT_EQ(kept.front(), 0u);
            EXPECT_EQ(kept.back(), n - 1);
            EXPECT_TRUE(std::ranges::is_sorted(kept));
        }
    }
}

TEST(SimplifyDatabase, MatchesSimplifyingEachTrajectory) {
    const auto db = make_random_db({30, 12, 40});
    const std::vector<std::size_t> budgets{5, 4, 10};

    for (const auto algorithm : kAlgorithms) {
        const auto simplification = selection::simplify_database(db, algorithm, budgets);

        for (std::size_t i = 0; i < db.size(); ++i)
            EXPECT_TRUE(std::ranges::equal(
                simplification.trajectory(i),
                selection::simplify_trajectory(db.trajectory(i), algorithm, budgets[i])));
    }
}

TEST(SimplifyDatabase, ThrowsWithoutOneBudgetPerTrajectory) {
    const auto db = make_random_db({5, 5});
    const std::vector<std::size_t> budgets{3};

    EXPECT_THROW(selection::simplify_database(db, selection::AlgorithmType::RDP, budgets),
                 std::invalid_argument);
}
