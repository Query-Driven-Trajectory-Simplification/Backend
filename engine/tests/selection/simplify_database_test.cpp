#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/selection/rdp.hpp"
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
                                                    selection::AlgorithmType::SED,
                                                    selection::AlgorithmType::UNIFORM};

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

TEST(GlobalRdp, EachAddedPointHasTheLargestRemainingPedAcrossTheDatabase) {
    const auto db = make_random_db({2, 5, 20, 50, 8});
    auto previous = core::endpoints_only(db);
    for (std::size_t budget = 2 * db.size() + 1; budget <= db.num_points(); ++budget) {
        const auto result = selection::rdp::simplify_database(db, budget);
        ASSERT_EQ(result.num_points(), budget);
        const auto kept = core::to_kept(result);
        double largest = -1.0, added_error = -1.0;
        std::size_t additions = 0;
        for (std::size_t i = 0; i < db.size(); ++i) {
            const auto trajectory = db.trajectory(i);
            EXPECT_EQ(kept[i].front(), 0u);
            EXPECT_EQ(kept[i].back(), trajectory.size() - 1);
            EXPECT_TRUE(std::ranges::is_sorted(kept[i]));
            EXPECT_EQ(std::ranges::adjacent_find(kept[i]), kept[i].end());
            EXPECT_TRUE(std::ranges::includes(kept[i], previous[i]));
            // Independent reference: measure every omitted point against its enclosing segment.
            for (std::size_t segment = 1; segment < previous[i].size(); ++segment) {
                const auto a = previous[i][segment - 1], b = previous[i][segment];
                const double dx = trajectory.x[b] - trajectory.x[a];
                const double dy = trajectory.y[b] - trajectory.y[a];
                for (std::size_t p = a + 1; p < b; ++p) {
                    const double u = dx * dx + dy * dy == 0.0
                                         ? 0.0
                                         : std::clamp(((trajectory.x[p] - trajectory.x[a]) * dx +
                                                       (trajectory.y[p] - trajectory.y[a]) * dy) /
                                                          (dx * dx + dy * dy),
                                                      0.0, 1.0);
                    const double error = std::hypot(trajectory.x[p] - trajectory.x[a] - u * dx,
                                                    trajectory.y[p] - trajectory.y[a] - u * dy);
                    largest = std::max(largest, error);
                    if (std::ranges::binary_search(kept[i], p)) {
                        added_error = error;
                        ++additions;
                    }
                }
            }
        }
        EXPECT_EQ(additions, 1u);
        EXPECT_NEAR(added_error, largest, 1e-9);
        previous = kept;
    }
}

TEST(GlobalRdp, ReservesEndpointsAndRejectsInfeasibleBudgets) {
    const auto db = make_random_db({2, 5, 20});
    EXPECT_EQ(core::to_kept(selection::rdp::simplify_database(db, 6)), core::endpoints_only(db));
    const auto full = selection::rdp::simplify_database(db, db.num_points());
    EXPECT_EQ(full.num_points(), db.num_points());
    for (std::size_t i = 0; i < db.size(); ++i)
        EXPECT_EQ(full.trajectory(i).size(), db.trajectory(i).size());
    EXPECT_THROW(selection::rdp::simplify_database(db, 5), std::invalid_argument);
    EXPECT_THROW(selection::rdp::simplify_database(db, db.num_points() + 1), std::invalid_argument);
    EXPECT_THROW(
        selection::simplify_trajectory(db.trajectory(0), selection::AlgorithmType::RDP_GLOBAL, 2),
        std::invalid_argument);
}

TEST(GlobalRdp, HandlesTiesAndTrajectoriesShorterThanTwoPoints) {
    const auto db = make_random_db({0, 1, 2, 8});
    const auto floor = core::endpoints_only(db);
    EXPECT_EQ(core::to_kept(selection::rdp::simplify_database(db, 5)), floor);
    EXPECT_EQ(selection::rdp::simplify_database(db, db.num_points()).num_points(), db.num_points());
    EXPECT_EQ(selection::rdp::simplify_database(make_random_db({}), 0).num_points(), 0u);
    // Identical trajectories create exact cross-trajectory ties.
    const auto one = make_random_db({8});
    const auto duplicated = core::subset(one, std::vector<std::size_t>{0, 0});
    const auto first = selection::rdp::simplify_database(duplicated, 5);
    EXPECT_EQ(first.trajectory(0).size(), 3u);
    EXPECT_EQ(first.trajectory(1).size(), 2u);
    EXPECT_EQ(core::to_kept(first),
              core::to_kept(selection::rdp::simplify_database(duplicated, 5)));
}
