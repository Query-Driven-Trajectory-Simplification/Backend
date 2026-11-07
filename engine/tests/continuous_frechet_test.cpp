#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include "backend/engine/core/projection.hpp"
#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/evaluator/join.hpp"
#include "backend/engine/evaluator/route_clustering.hpp"
#include "backend/engine/query/frechet_helper.hpp"
#include "backend/engine/query/join.hpp"
#include "backend/engine/query/knn.hpp"
#include "backend/engine/query/route_clustering.hpp"

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

double continuous(const core::TrajectoryDb &db, std::size_t a, std::size_t b) {
    return query::continuous_frechet_distance(db.trajectory(a), db.trajectory(b), 0.0, 1e7);
}

std::vector<Line> random_lines(std::mt19937_64 &rng, std::size_t count, std::size_t max_len) {
    std::uniform_real_distribution<double> start(-300.0, 300.0);
    std::normal_distribution<double> step(40.0, 60.0);
    std::uniform_int_distribution<std::size_t> length(2, max_len);
    std::vector<Line> lines(count);
    for (auto &line : lines) {
        double x = start(rng), y = start(rng);
        for (std::size_t k = 0, n = length(rng); k < n; ++k) {
            line.emplace_back(x, y);
            x += step(rng);
            y += step(rng) * 0.5;
        }
    }
    return lines;
}

} // namespace

TEST(ContinuousFrechet, AddingAVertexOnAnEdgeChangesNothing) {
    // Discrete Fréchet jumps to half the segment once one side gains its true midpoint
    const auto db = make_db({{{0, 0}, {1000, 0}}, {{0, 0}, {500, 0}, {1000, 0}}});
    EXPECT_NEAR(continuous(db, 0, 1), 0.0, 1e-2);
    EXPECT_NEAR(query::frechet_distance(db.trajectory(0), db.trajectory(1)), 500.0, 1e-2);
    EXPECT_TRUE(query::continuous_frechet_within(db.trajectory(0), db.trajectory(1), 1.0));
    EXPECT_FALSE(query::frechet_within(db.trajectory(0), db.trajectory(1), 1.0));
}

TEST(ContinuousFrechet, KnownDistances) {
    // A tent of height 1 against its base, parallel segments 3 apart, and a backtrack of 0.5
    const auto db = make_db({{{0, 0}, {1, 1}, {2, 0}},
                             {{0, 0}, {2, 0}},
                             {{0, 3}, {2, 3}},
                             {{0, 0}, {1.5, 0}, {1, 0}, {2, 0}}});
    EXPECT_NEAR(continuous(db, 0, 1), 1.0, 1e-2);
    EXPECT_NEAR(continuous(db, 1, 2), 3.0, 1e-2);
    EXPECT_NEAR(continuous(db, 3, 1), 0.25, 1e-2);
}

TEST(ContinuousFrechet, NeverAboveDiscreteAndInvariantUnderSubdivision) {
    std::mt19937_64 rng(5);
    const auto lines = random_lines(rng, 120, 12);
    std::vector<Line> subdivided;
    for (const auto &line : lines) {
        Line s{line.front()};
        for (std::size_t k = 1; k < line.size(); ++k) {
            const auto [ax, ay] = line[k - 1];
            const auto [bx, by] = line[k];
            s.emplace_back(0.5 * (ax + bx), 0.5 * (ay + by));
            s.push_back(line[k]);
        }
        subdivided.push_back(s);
    }
    std::vector<Line> both = lines;
    both.insert(both.end(), subdivided.begin(), subdivided.end());
    const auto db = make_db(both);
    const std::size_t n = lines.size();
    for (std::size_t a = 0; a + 1 < n; a += 2) {
        const std::size_t b = a + 1;
        const double c = continuous(db, a, b);
        EXPECT_LE(c, query::frechet_distance(db.trajectory(a), db.trajectory(b)) + 1e-2);
        EXPECT_NEAR(c, continuous(db, a + n, b + n), 2e-2);
        EXPECT_NEAR(c, continuous(db, b, a), 2e-2);
    }
}

TEST(ContinuousFrechet, DecisionAgreesWithDistance) {
    std::mt19937_64 rng(9);
    const auto db = make_db(random_lines(rng, 60, 10));
    for (std::size_t a = 0; a + 1 < db.size(); ++a) {
        const double d = continuous(db, a, a + 1);
        EXPECT_TRUE(
            query::continuous_frechet_within(db.trajectory(a), db.trajectory(a + 1), d + 0.01));
        if (d > 0.02)
            EXPECT_FALSE(
                query::continuous_frechet_within(db.trajectory(a), db.trajectory(a + 1), d - 0.01));
    }
}

TEST(ContinuousFrechet, KnnMatchesBruteForce) {
    std::mt19937_64 rng(13);
    const auto db = make_db(random_lines(rng, 80, 10));
    const auto queries = make_db(random_lines(rng, 12, 10));
    for (std::size_t q = 0; q < queries.size(); ++q) {
        std::vector<std::pair<double, std::size_t>> all;
        for (std::size_t i = 0; i < db.size(); ++i)
            all.emplace_back(query::continuous_frechet_distance(queries.trajectory(q),
                                                                db.trajectory(i), 0.0, 1e7),
                             i);
        std::sort(all.begin(), all.end());
        const auto got =
            query::knn(db, queries.trajectory(q), 5, query::FrechetVariant::Continuous);
        ASSERT_EQ(got.size(), 5u);
        // Distances are exact to 1 mm, so compare the distance of each rank
        for (std::size_t r = 0; r < 5; ++r) {
            const double d = query::continuous_frechet_distance(queries.trajectory(q),
                                                                db.trajectory(got[r]), 0.0, 1e7);
            EXPECT_NEAR(d, all[r].first, 5e-3);
        }
    }
}

TEST(ContinuousFrechet, KnnOrdersEqualDistancesByIndex) {
    // Copies of one line at the same distance, plus a nearer and a farther line
    const Line line{{0, 10}, {500, 10}, {1000, 10}};
    const auto db = make_db({line, {{0, 50}, {1000, 50}}, line, {{0, 2}, {1000, 2}}, line});
    const auto query = make_db({{{0, 0}, {1000, 0}}});
    const auto got = query::knn(db, query.trajectory(0), 4, query::FrechetVariant::Continuous);
    EXPECT_EQ(got, (std::vector<std::size_t>{3, 0, 2, 4}));
}

TEST(ContinuousFrechet, KnnMatchesBruteForceOnDenseTrajectories) {
    // Long, densely sampled lines take the discrete upper-bound path
    std::mt19937_64 rng(17);
    const auto db = make_db(random_lines(rng, 60, 80));
    const auto queries = make_db(random_lines(rng, 8, 80));
    for (std::size_t q = 0; q < queries.size(); ++q) {
        std::vector<double> all;
        for (std::size_t i = 0; i < db.size(); ++i)
            all.push_back(query::continuous_frechet_distance(queries.trajectory(q),
                                                             db.trajectory(i), 0.0, 1e7));
        std::ranges::sort(all);
        const auto got =
            query::knn(db, queries.trajectory(q), 7, query::FrechetVariant::Continuous);
        ASSERT_EQ(got.size(), 7u);
        for (std::size_t r = 0; r < got.size(); ++r)
            EXPECT_NEAR(query::continuous_frechet_distance(queries.trajectory(q),
                                                           db.trajectory(got[r]), 0.0, 1e7),
                        all[r], 5e-3);
    }
}

TEST(ContinuousFrechet, SelfJoinMatchesBruteForceAndScoresOriginalPerfectly) {
    std::mt19937_64 rng(21);
    const auto db = make_db(random_lines(rng, 60, 8));
    const double eps = 150.0;
    const auto workload = query::generate_self_join(db, eps, query::FrechetVariant::Continuous);
    std::size_t expected = 0, got = 0;
    for (std::size_t i = 0; i < db.size(); ++i) {
        got += workload.entries[i].reference.size();
        for (std::size_t j = i + 1; j < db.size(); ++j)
            expected += query::continuous_frechet_within(db.trajectory(i), db.trajectory(j), eps);
    }
    EXPECT_EQ(got, expected);
    const auto scores = evaluator::evaluate_join(db, workload);
    if (expected > 0)
        EXPECT_DOUBLE_EQ(scores.f1, 1.0);
}

TEST(ContinuousFrechet, ClusteringReusesOnlyASelfJoinOfTheSameVariant) {
    // Three copies of a line and a fourth line that matches them only under continuous distance
    const Line line{{0, 0}, {1000, 0}};
    const Line subdivided{{0, 0}, {500, 0}, {1000, 0}};
    const auto db = make_db({line, line, line, subdivided});
    const auto discrete_join = query::generate_self_join(db, 100.0);
    const query::RouteClusteringConfig config{.enabled = true,
                                              .eps_m = 100.0,
                                              .min_size = 4,
                                              .frechet = query::FrechetVariant::Continuous};
    const auto workload = query::generate_route_clustering(db, config, &discrete_join);
    ASSERT_EQ(workload.reference.members.size(), 1u);
    EXPECT_EQ(workload.reference.members[0].size(), 4u);
    EXPECT_DOUBLE_EQ(evaluator::evaluate_route_clustering(db, workload).score, 1.0);
}

TEST(ContinuousFrechet, KnnFindsAnExactMatchFarFromTheOrigin) {
    const auto db = make_db({{{-10000.123, 12345.678}, {0.01, 0.02}}});
    EXPECT_EQ(query::knn(db, db.trajectory(0), 1, query::FrechetVariant::Continuous),
              (std::vector<std::size_t>{0}));
}

TEST(ContinuousFrechet, KnnTiesFromDifferentGeometriesAreConsistentAcrossK) {
    // A tent of height 10 and a parallel line 10 m away are equally far from the query
    const Line tent{{0, 0}, {500, 10}, {1000, 0}};
    const Line parallel{{0, 10}, {1000, 10}};
    const auto query = make_db({{{0, 0}, {1000, 0}}});
    for (const auto &[lines, first] : {std::pair{std::vector<Line>{tent, parallel}, 0u},
                                       std::pair{std::vector<Line>{parallel, tent}, 0u}}) {
        const auto db = make_db(lines);
        const auto k = query::FrechetVariant::Continuous;
        EXPECT_EQ(query::knn(db, query.trajectory(0), 1, k), (std::vector<std::size_t>{first}));
        EXPECT_EQ(query::knn(db, query.trajectory(0), 2, k), (std::vector<std::size_t>{0, 1}));
    }
}

// A view straight from local coordinates, without the lat/lon round trip
struct RawLine {
    std::vector<std::int64_t> t;
    std::vector<double> x, y;
    explicit RawLine(const Line &line) {
        for (std::size_t k = 0; k < line.size(); ++k) {
            t.push_back(static_cast<std::int64_t>(k));
            x.push_back(line[k].first);
            y.push_back(line[k].second);
        }
    }
    core::TrajectoryView view() const { return {0, t, {}, {}, x, y}; }
};

TEST(ContinuousFrechet, LongSegmentWithAVertexOnItIsAtDistanceZeroExactCoordinates) {
    const double bx = -81677.87045, by = -85151.97823;
    for (const double f : {0.75641311848, 0.5, 0.1, 0.999, 0.333333333}) {
        const RawLine a({{0, 0}, {bx, by}});
        const RawLine b({{0, 0}, {f * bx, f * by}, {bx, by}});
        EXPECT_LT(query::continuous_frechet_distance(a.view(), b.view(), 0.0, 1e6), 1e-3) << f;
        EXPECT_TRUE(query::continuous_frechet_within(a.view(), b.view(), 1e-3)) << f;
        EXPECT_TRUE(query::continuous_frechet_within(b.view(), a.view(), 1e-3)) << f;
    }
}

TEST(ContinuousFrechet, LongSegmentWithAVertexOnItIsAtDistanceZero) {
    // Built in projected coordinates, so the inserted vertex lies on the segment
    const auto ends = make_db({{{0, 0}, {-81677.87045, -85151.97823}}}).trajectory(0);
    const double ax = ends.x[0], ay = ends.y[0], bx = ends.x[1], by = ends.y[1];
    for (const double f : {0.75641311848, 0.5, 0.1, 0.999}) {
        const auto db = make_db(
            {{{ax, ay}, {bx, by}}, {{ax, ay}, {ax + f * (bx - ax), ay + f * (by - ay)}, {bx, by}}});
        EXPECT_LT(continuous(db, 0, 1), 1e-3) << f;
        EXPECT_TRUE(query::continuous_frechet_within(db.trajectory(0), db.trajectory(1), 1e-3))
            << f;
    }
}

TEST(ContinuousFrechet, NearTangentSegmentIsJustWithinItsDistance) {
    // The vertex (500, d) touches the disc of radius d around the segment's closest point
    const auto db = make_db({{{0, 0}, {1000, 0}}, {{0, 0}, {500, 37.5}, {1000, 0}}});
    EXPECT_NEAR(continuous(db, 0, 1), 37.5, 2e-3);
    EXPECT_TRUE(query::continuous_frechet_within(db.trajectory(0), db.trajectory(1), 37.5 + 1e-3));
    EXPECT_FALSE(query::continuous_frechet_within(db.trajectory(0), db.trajectory(1), 37.5 - 1e-2));
}

TEST(ContinuousFrechet, AgreesWithDiscreteDistanceOnDenselySubdividedLines) {
    // An independent check: on lines subdivided into steps of at most s, the discrete distance
    // is at least the continuous one and at most s above it
    std::mt19937_64 rng(31);
    const auto lines = random_lines(rng, 40, 6);
    const double step = 2.0;
    std::vector<Line> dense;
    for (const auto &line : lines) {
        Line d{line.front()};
        for (std::size_t k = 1; k < line.size(); ++k) {
            const auto [ax, ay] = line[k - 1];
            const auto [bx, by] = line[k];
            const auto parts =
                static_cast<std::size_t>(std::ceil(std::hypot(bx - ax, by - ay) / step));
            for (std::size_t s = 1; s <= parts; ++s) {
                const double f = static_cast<double>(s) / static_cast<double>(parts);
                d.emplace_back(ax + f * (bx - ax), ay + f * (by - ay));
            }
        }
        dense.push_back(d);
    }
    const auto sparse_db = make_db(lines);
    const auto dense_db = make_db(dense);
    for (std::size_t a = 0; a + 1 < lines.size(); a += 2) {
        const double c = continuous(sparse_db, a, a + 1);
        const double d =
            query::frechet_distance(dense_db.trajectory(a), dense_db.trajectory(a + 1));
        EXPECT_LE(c, d + 2e-3);
        EXPECT_GE(c, d - step - 1e-2);
    }
}
