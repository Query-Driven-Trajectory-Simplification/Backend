#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <random>
#include <stdexcept>
#include <vector>

#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/selection/bottom_up.hpp"
#include "backend/engine/selection/distance_functions.hpp"
#include "backend/engine/selection/sed.hpp"

namespace {

using selection::bottom_up::Measure;
using Indices = std::vector<std::size_t>;

constexpr Measure kMeasures[] = {Measure::Sed, Measure::Ped, Measure::Dad};

struct Trajectory {
    std::vector<std::int64_t> t;
    std::vector<double> x;
    std::vector<double> y;

    Trajectory(std::vector<double> xs, std::vector<double> ys, std::vector<std::int64_t> ts = {})
        : t(std::move(ts)), x(std::move(xs)), y(std::move(ys)) {
        if (t.empty())
            for (std::size_t i = 0; i < x.size(); ++i)
                t.push_back(static_cast<std::int64_t>(i));
    }

    core::TrajectoryView view() const { return {0, t, {}, {}, x, y}; }
};

Trajectory random_walk(std::mt19937_64 &rng, std::size_t n) {
    std::uniform_real_distribution<double> step(-10.0, 10.0);
    std::uniform_int_distribution<std::int64_t> gap(1, 5);
    std::vector<double> x{0.0}, y{0.0};
    std::vector<std::int64_t> t{0};
    for (std::size_t i = 1; i < n; ++i) {
        x.push_back(x.back() + step(rng));
        y.push_back(y.back() + step(rng));
        t.push_back(t.back() + gap(rng));
    }
    return {x, y, t};
}

double error(const core::TrajectoryView &trajectory, const Indices &kept, Measure measure) {
    double worst = 0.0;
    for (std::size_t s = 0; s + 1 < kept.size(); ++s)
        worst = std::max(
            worst, selection::bottom_up::span_error(trajectory, kept[s], kept[s + 1], measure));
    return worst;
}

core::TrajectoryDb random_db(std::mt19937_64 &rng, const std::vector<std::size_t> &sizes) {
    std::uniform_real_distribution<double> offset(-0.01, 0.01);
    std::uniform_int_distribution<std::int64_t> gap(1, 5);
    std::vector<std::uint64_t> offsets{0};
    std::vector<std::int32_t> ids;
    std::vector<std::int64_t> t;
    std::vector<double> lat;
    std::vector<double> lon;
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        ids.push_back(static_cast<std::int32_t>(i));
        std::int64_t time = 0;
        for (std::size_t k = 0; k < sizes[i]; ++k) {
            t.push_back(time);
            time += gap(rng);
            lat.push_back(39.95 + offset(rng));
            lon.push_back(116.35 + offset(rng));
        }
        offsets.push_back(t.size());
    }
    return {std::move(offsets), std::move(ids), std::move(t), std::move(lat), std::move(lon)};
}

std::vector<Indices> brute_force_global(const core::TrajectoryDb &db, std::size_t budget,
                                        Measure measure) {
    std::vector<std::vector<char>> kept(db.size());
    std::size_t count = 0;
    for (std::size_t i = 0; i < db.size(); ++i) {
        kept[i].assign(db.trajectory(i).size(), 1);
        count += kept[i].size();
    }
    while (count > budget) {
        double best = std::numeric_limits<double>::infinity();
        std::size_t best_i = 0, best_k = 0;
        for (std::size_t i = 0; i < db.size(); ++i) {
            const auto view = db.trajectory(i);
            for (std::size_t k = 1; k + 1 < view.size(); ++k) {
                if (!kept[i][k])
                    continue;
                std::size_t prev = k - 1, next = k + 1;
                while (!kept[i][prev])
                    --prev;
                while (!kept[i][next])
                    ++next;
                const double cost = selection::bottom_up::span_error(view, prev, next, measure);
                if (cost < best) {
                    best = cost;
                    best_i = i;
                    best_k = k;
                }
            }
        }
        kept[best_i][best_k] = 0;
        --count;
    }
    std::vector<Indices> result(db.size());
    for (std::size_t i = 0; i < db.size(); ++i)
        for (std::size_t k = 0; k < kept[i].size(); ++k)
            if (kept[i][k])
                result[i].push_back(k);
    return result;
}

std::vector<Indices> as_indices(const core::Simplification &simplification) {
    std::vector<Indices> result(simplification.size());
    for (std::size_t i = 0; i < simplification.size(); ++i) {
        const auto kept = simplification.trajectory(i);
        result[i].assign(kept.begin(), kept.end());
    }
    return result;
}

} // namespace

TEST(BottomUp, SedMeasureIsTheSedSimplifier) {
    std::mt19937_64 rng(3);
    for (int trial = 0; trial < 20; ++trial) {
        const auto traj = random_walk(rng, 40);
        EXPECT_EQ(selection::bottom_up::sequence(traj.view(), 40, Measure::Sed),
                  selection::sed::sequence(traj.view(), 40));
        // The span error is the largest synchronous distance in the span
        double worst = 0.0;
        for (std::size_t k = 1; k < 39; ++k)
            worst = std::max(worst, selection::sed::synchronous_distance(traj.view(), 0, 39, k));
        EXPECT_NEAR(selection::bottom_up::span_error(traj.view(), 0, 39, Measure::Sed), worst,
                    1e-9);
    }
}

TEST(BottomUp, PedIsTheDistanceToTheSegment) {
    const Trajectory bump({0, 5, 10}, {0, 3, 0});
    EXPECT_DOUBLE_EQ(selection::bottom_up::span_error(bump.view(), 0, 2, Measure::Ped), 3.0);
    const Trajectory beyond({0, 14, 10}, {0, 2, 0});
    EXPECT_DOUBLE_EQ(selection::bottom_up::span_error(beyond.view(), 0, 2, Measure::Ped),
                     std::sqrt(20.0));
    const Trajectory loop({0, 5, 0}, {0, 5, 0});
    EXPECT_DOUBLE_EQ(selection::bottom_up::span_error(loop.view(), 0, 2, Measure::Ped),
                     std::sqrt(50.0));
}

TEST(BottomUp, PedMatchesTheRdpDistance) {
    std::mt19937_64 rng(13);
    for (int trial = 0; trial < 20; ++trial) {
        const auto traj = random_walk(rng, 30);
        const auto view = traj.view();
        double worst = 0.0;
        for (std::size_t k = 1; k < 29; ++k)
            worst = std::max(
                worst, selection::distance::squared_segment_distance(
                           view.x[k], view.y[k], view.x[0], view.y[0], view.x[29], view.y[29]));
        EXPECT_DOUBLE_EQ(selection::bottom_up::span_error(view, 0, 29, Measure::Ped),
                         std::sqrt(worst));
    }
}

TEST(BottomUp, DadIsTheLargestHeadingChange) {
    // Up 45 degrees then down 45 degrees against a flat kept segment
    const Trajectory roof({0, 1, 2}, {0, 1, 0});
    EXPECT_NEAR(selection::bottom_up::span_error(roof.view(), 0, 2, Measure::Dad),
                std::numbers::pi / 4, 1e-12);
    // Turning back: 180 degrees
    const Trajectory back({0, 2, 1}, {0, 0, 0});
    EXPECT_NEAR(selection::bottom_up::span_error(back.view(), 0, 2, Measure::Dad), std::numbers::pi,
                1e-12);
    // Across 0 / 2 pi the circular difference is taken
    const Trajectory wrap({0, 1, 2}, {0, -0.1, 0});
    EXPECT_NEAR(selection::bottom_up::span_error(wrap.view(), 0, 2, Measure::Dad),
                std::atan2(0.1, 1.0), 1e-12);
}

TEST(BottomUp, StraightConstantSpeedHasNoErrorUnderAnyMeasure) {
    const Trajectory line({0, 1, 2, 3, 4, 5}, {0, 2, 4, 6, 8, 10});
    for (const auto measure : kMeasures) {
        EXPECT_NEAR(selection::bottom_up::span_error(line.view(), 0, 5, measure), 0.0, 1e-12);
        EXPECT_EQ(selection::bottom_up::simplify(line.view(), 2, measure), (Indices{0, 5}));
    }
}

TEST(BottomUp, OnlySedSeesASpeedChangeOnTheSamePath) {
    // Same straight path, but most of the distance is covered in the last time step
    const Trajectory jump({0, 1, 2, 10}, {0, 0, 0, 0});
    EXPECT_GT(selection::bottom_up::span_error(jump.view(), 0, 3, Measure::Sed), 1.0);
    EXPECT_DOUBLE_EQ(selection::bottom_up::span_error(jump.view(), 0, 3, Measure::Ped), 0.0);
    EXPECT_DOUBLE_EQ(selection::bottom_up::span_error(jump.view(), 0, 3, Measure::Dad), 0.0);
}

TEST(BottomUp, MeasuresKeepDifferentPoints) {
    // Point 1 is a small sideways bump at the right time; point 3 is on the line but far ahead
    // of schedule. PED keeps the bump, SED the point off schedule
    const Trajectory traj({0, 2.5, 5, 9.5, 10}, {0, 1, 0, 0, 0}, {0, 1, 2, 3, 4});
    EXPECT_EQ(selection::bottom_up::simplify(traj.view(), 3, Measure::Ped), (Indices{0, 1, 4}));
    EXPECT_EQ(selection::bottom_up::simplify(traj.view(), 3, Measure::Sed), (Indices{0, 3, 4}));
}

TEST(BottomUp, EachRemovalIsTheCheapest) {
    std::mt19937_64 rng(11);
    for (const auto measure : kMeasures)
        for (int trial = 0; trial < 10; ++trial) {
            const auto traj = random_walk(rng, 30);
            const auto view = traj.view();
            const auto order = selection::bottom_up::sequence(view, 30, measure);
            ASSERT_EQ(order.size(), 30u);
            // Replay the removals: the reverse of everything after the endpoints
            std::vector<char> kept(30, 1);
            for (auto it = order.rbegin(); it + 2 != order.rend(); ++it) {
                const auto removed = *it;
                auto prev_of = [&](std::size_t i) {
                    while (!kept[--i]) {
                    }
                    return i;
                };
                auto next_of = [&](std::size_t i) {
                    while (!kept[++i]) {
                    }
                    return i;
                };
                double cheapest = std::numeric_limits<double>::infinity();
                for (std::size_t i = 1; i + 1 < 30; ++i)
                    if (kept[i])
                        cheapest = std::min(cheapest, selection::bottom_up::span_error(
                                                          view, prev_of(i), next_of(i), measure));
                EXPECT_LE(selection::bottom_up::span_error(view, prev_of(removed), next_of(removed),
                                                           measure),
                          cheapest + 1e-9);
                kept[removed] = 0;
            }
        }
}

TEST(BottomUp, SequencePrefixesAreEachBudget) {
    std::mt19937_64 rng(5);
    for (const auto measure : kMeasures)
        for (int trial = 0; trial < 5; ++trial) {
            const auto traj = random_walk(rng, 25);
            const auto order = selection::bottom_up::sequence(traj.view(), 25, measure);
            for (std::size_t b = 2; b <= 25; ++b) {
                Indices prefix(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(b));
                std::ranges::sort(prefix);
                EXPECT_EQ(prefix, selection::bottom_up::simplify(traj.view(), b, measure));
            }
        }
}

TEST(BottomUp, AllPointsGiveZeroAndNoBudgetGivesEndpoints) {
    std::mt19937_64 rng(8);
    for (const auto measure : kMeasures) {
        const auto traj = random_walk(rng, 30);
        Indices all(30);
        for (std::size_t i = 0; i < 30; ++i)
            all[i] = i;
        EXPECT_EQ(selection::bottom_up::simplify(traj.view(), 30, measure), all);
        EXPECT_DOUBLE_EQ(error(traj.view(), all, measure), 0.0);
        EXPECT_EQ(selection::bottom_up::simplify(traj.view(), 0, measure), (Indices{0, 29}));
    }
}

TEST(BottomUp, RejectsEmptySpans) {
    const Trajectory traj({0, 1, 2}, {0, 1, 0});
    EXPECT_THROW(selection::bottom_up::span_error(traj.view(), 1, 1, Measure::Ped),
                 std::invalid_argument);
    EXPECT_THROW(selection::bottom_up::span_error(traj.view(), 0, 3, Measure::Dad),
                 std::invalid_argument);
}

TEST(BottomUp, GlobalMatchesOneHeapOverTheWholeDatabase) {
    std::mt19937_64 rng(21);
    const auto db = random_db(rng, {12, 3, 20, 2, 15, 9});
    std::size_t endpoints = 0;
    for (std::size_t i = 0; i < db.size(); ++i)
        endpoints += std::min<std::size_t>(db.trajectory(i).size(), 2);
    for (const auto measure : {Measure::Sed, Measure::Dad})
        for (std::size_t budget = endpoints; budget <= db.num_points(); budget += 5)
            EXPECT_EQ(as_indices(selection::bottom_up::simplify_database(db, budget, measure)),
                      brute_force_global(db, budget, measure))
                << budget;
}

TEST(BottomUp, GlobalKeepsEachTrajectorysOwnBottomUpAndTheBudget) {
    std::mt19937_64 rng(22);
    const auto db = random_db(rng, {12, 3, 20, 2, 15, 9});
    std::size_t endpoints = 0;
    for (std::size_t i = 0; i < db.size(); ++i)
        endpoints += std::min<std::size_t>(db.trajectory(i).size(), 2);
    for (const auto measure : kMeasures)
        for (std::size_t budget = endpoints; budget <= db.num_points(); budget += 3) {
            const auto kept =
                as_indices(selection::bottom_up::simplify_database(db, budget, measure));
            std::size_t total = 0;
            for (std::size_t i = 0; i < db.size(); ++i) {
                total += kept[i].size();
                EXPECT_EQ(kept[i],
                          selection::bottom_up::simplify(db.trajectory(i), kept[i].size(), measure))
                    << budget << " " << i;
            }
            EXPECT_EQ(total, budget);
        }
}

TEST(BottomUp, GlobalKeepsEndpointsAtTheLowestBudgetAndEverythingAtTheHighest) {
    std::mt19937_64 rng(4);
    const auto db = random_db(rng, {10, 2, 7});
    const auto low = as_indices(selection::bottom_up::simplify_database(db, 6, Measure::Ped));
    EXPECT_EQ(low, (std::vector<Indices>{{0, 9}, {0, 1}, {0, 6}}));
    const auto all =
        as_indices(selection::bottom_up::simplify_database(db, db.num_points(), Measure::Ped));
    for (std::size_t i = 0; i < db.size(); ++i)
        EXPECT_EQ(all[i].size(), db.trajectory(i).size());
    EXPECT_THROW(selection::bottom_up::simplify_database(db, 5, Measure::Ped),
                 std::invalid_argument);
    EXPECT_THROW(selection::bottom_up::simplify_database(db, db.num_points() + 1, Measure::Ped),
                 std::invalid_argument);
}
