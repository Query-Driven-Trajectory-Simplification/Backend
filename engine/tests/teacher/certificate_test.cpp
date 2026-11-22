#include <gtest/gtest.h>

#include "backend/engine/core/projection.hpp"
#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/query/frechet_helper.hpp"
#include "backend/engine/teacher/objective.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <utility>
#include <vector>

namespace {

using Line = std::vector<std::pair<double, double>>; // (x, y) in local metres

core::TrajectoryDb make_db(const std::vector<Line> &lines) {
    std::vector<std::uint64_t> offsets{0};
    std::vector<std::int32_t> ids;
    std::vector<std::int64_t> t;
    std::vector<double> lat;
    std::vector<double> lon;
    for (std::size_t i = 0; i < lines.size(); ++i) {
        ids.push_back(static_cast<std::int32_t>(i));
        for (std::size_t k = 0; k < lines[i].size(); ++k) {
            const auto ll = core::to_lat_lon(lines[i][k].first, lines[i][k].second);
            t.push_back(static_cast<std::int64_t>(k));
            lat.push_back(ll.lat);
            lon.push_back(ll.lon);
        }
        offsets.push_back(t.size());
    }
    return {std::move(offsets), std::move(ids), std::move(t), std::move(lat), std::move(lon)};
}

// Random walks with turns, stops (repeated points) and backtracking
std::vector<Line> random_lines(std::mt19937_64 &rng, std::size_t count, std::size_t max_len) {
    std::uniform_real_distribution<double> start(-200.0, 200.0);
    std::normal_distribution<double> step(0.0, 50.0);
    std::uniform_int_distribution<std::size_t> length(2, max_len);
    std::bernoulli_distribution stop(0.1);
    std::vector<Line> lines(count);
    for (auto &line : lines) {
        double x = start(rng), y = start(rng);
        for (std::size_t k = 0, n = length(rng); k < n; ++k) {
            line.emplace_back(x, y);
            if (!stop(rng)) {
                x += step(rng) + 20.0;
                y += step(rng);
            }
        }
    }
    return lines;
}

// Sorted random subset of [0, n) with both endpoints
std::vector<std::size_t> random_kept(std::mt19937_64 &rng, std::size_t n) {
    std::bernoulli_distribution keep(std::uniform_real_distribution<double>(0.0, 0.6)(rng));
    std::vector<std::size_t> kept{0};
    for (std::size_t j = 1; j + 1 < n; ++j)
        if (keep(rng))
            kept.push_back(j);
    kept.push_back(n - 1);
    return kept;
}

TEST(SimplificationBound, IsZeroForTheTrajectoryItself) {
    const auto db = make_db({{{0, 0}, {100, 0}, {100, 100}, {0, 100}}});
    const std::vector<std::size_t> all{0, 1, 2, 3};
    const teacher::PreparedTrajectory kept(db.trajectory(0), all);
    EXPECT_NEAR(teacher::simplification_bound(db.trajectory(0), kept.view()), 0.0, 1e-9);
}

TEST(SimplificationBound, IsInfiniteWithoutASegment) {
    const auto db = make_db({{{0, 0}, {100, 0}, {200, 0}}});
    const std::vector<std::size_t> one{0};
    const teacher::PreparedTrajectory kept(db.trajectory(0), one);
    EXPECT_TRUE(std::isinf(teacher::simplification_bound(db.trajectory(0), kept.view())));
}

TEST(SimplificationBound, IsInfiniteForPointsNotInTheOriginal) {
    const auto db = make_db({{{0, 0}, {100, 0}, {200, 0}}, {{0, 0}, {50, 50}, {200, 0}}});
    const std::vector<std::size_t> all{0, 1, 2};
    const teacher::PreparedTrajectory other(db.trajectory(1), all);
    EXPECT_TRUE(std::isinf(teacher::simplification_bound(db.trajectory(0), other.view())));
}

TEST(SimplificationBound, MatchesTheDetourOfAnExcursion) {
    // Dropping the peak of a 50 m spike leaves 50 m as the continuous Fréchet distance
    const auto db = make_db({{{0, 0}, {100, 0}, {100, 50}, {100, 0}, {200, 0}}});
    const std::vector<std::size_t> ends{0, 4};
    const teacher::PreparedTrajectory kept(db.trajectory(0), ends);
    const double bound = teacher::simplification_bound(db.trajectory(0), kept.view());
    EXPECT_GE(bound, 50.0 - 1e-6);
    EXPECT_TRUE(query::continuous_frechet_within(db.trajectory(0), kept.view(), bound));
}

TEST(SimplificationBound, NeverBelowTheContinuousFrechetDistance) {
    std::mt19937_64 rng(7);
    const auto db = make_db(random_lines(rng, 200, 40));
    for (std::size_t i = 0; i < db.size(); ++i) {
        const auto original = db.trajectory(i);
        for (int trial = 0; trial < 5; ++trial) {
            const auto indices = random_kept(rng, original.size());
            const teacher::PreparedTrajectory kept(original, indices);
            const double bound = teacher::simplification_bound(original, kept.view());
            ASSERT_TRUE(std::isfinite(bound));
            EXPECT_TRUE(query::continuous_frechet_within(original, kept.view(), bound))
                << "trajectory " << i << " trial " << trial << " bound " << bound;
        }
    }
}

// The certificate in Objective::held_out_match: d(P, C) <= discrete d(P, T) + bound(T, C)
TEST(SimplificationBound, CertificateNeverBelowTheContinuousFrechetDistance) {
    std::mt19937_64 rng(11);
    const auto db = make_db(random_lines(rng, 60, 30));
    std::vector<double> prev, curr;
    for (std::size_t a = 0; a < db.size(); ++a) {
        for (std::size_t b = 0; b < db.size(); ++b) {
            if (a == b)
                continue;
            const auto partner = db.trajectory(a);
            const auto original = db.trajectory(b);
            const double pair = std::sqrt(query::get_squared_frechet_distance(
                partner, original, std::numeric_limits<double>::infinity(), prev, curr));
            EXPECT_TRUE(query::continuous_frechet_within(partner, original, pair));
            const auto indices = random_kept(rng, original.size());
            const teacher::PreparedTrajectory kept(original, indices);
            const double certificate = pair + teacher::simplification_bound(original, kept.view());
            EXPECT_TRUE(query::continuous_frechet_within(partner, kept.view(), certificate))
                << "pair " << a << ", " << b << " certificate " << certificate;
        }
    }
}

} // namespace
