#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "backend/engine/core/simplification.hpp"
#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/evaluator/join.hpp"
#include "backend/engine/query/join.hpp"
#include "backend/engine/query/frechet_helper.hpp"
#include "backend/engine/selection/selector.hpp"

namespace {

using Track = std::vector<std::pair<double, double>>; // (lat, lon) per point

core::TrajectoryDb make_db(const std::vector<Track> &tracks) {
    std::vector<std::uint64_t> offsets{0};
    std::vector<std::int32_t> ids;
    std::vector<std::int64_t> t;
    std::vector<double> lat;
    std::vector<double> lon;
    for (std::size_t i = 0; i < tracks.size(); ++i) {
        ids.push_back(static_cast<std::int32_t>(i));
        for (std::size_t k = 0; k < tracks[i].size(); ++k) {
            t.push_back(static_cast<std::int64_t>(k));
            lat.push_back(tracks[i][k].first);
            lon.push_back(tracks[i][k].second);
        }
        offsets.push_back(t.size());
    }
    return {std::move(offsets), std::move(ids), std::move(t), std::move(lat), std::move(lon)};
}

// Random walks starting in a small area, so many are within a few hundred metres of each other
std::vector<Track> random_tracks(std::mt19937_64 &rng, std::size_t count) {
    std::uniform_real_distribution<double> start_lat(39.90, 39.92);
    std::uniform_real_distribution<double> start_lon(116.30, 116.32);
    std::uniform_int_distribution<std::size_t> length(2, 40);
    std::normal_distribution<double> step(0.0, 0.0005);
    std::vector<Track> tracks(count);
    for (auto &track : tracks) {
        double la = start_lat(rng);
        double lo = start_lon(rng);
        const std::size_t n = length(rng);
        for (std::size_t k = 0; k < n; ++k) {
            track.emplace_back(la, lo);
            la += step(rng);
            lo += step(rng);
        }
    }
    return tracks;
}

double frechet(const core::TrajectoryView &a, const core::TrajectoryView &b) {
    std::vector<std::vector<double>> dp(a.size(), std::vector<double>(b.size()));
    for (std::size_t i = 0; i < a.size(); ++i) {
        for (std::size_t j = 0; j < b.size(); ++j) {
            const double d = std::hypot(a.x[i] - b.x[j], a.y[i] - b.y[j]);
            if (i == 0 && j == 0)
                dp[i][j] = d;
            else if (i == 0)
                dp[i][j] = std::max(d, dp[i][j - 1]);
            else if (j == 0)
                dp[i][j] = std::max(d, dp[i - 1][j]);
            else
                dp[i][j] = std::max(d, std::min({dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]}));
        }
    }
    return dp.back().back();
}

} // namespace

TEST(Join, FrechetHelpersMatchBruteForce) {
    std::mt19937_64 rng(5);
    const auto db = make_db(random_tracks(rng, 60));
    for (std::size_t a = 0; a < db.size(); a += 7) {
        for (std::size_t b = 0; b < db.size(); ++b) {
            const double expected = frechet(db.trajectory(a), db.trajectory(b));
            EXPECT_NEAR(query::frechet_distance(db.trajectory(a), db.trajectory(b)), expected,
                        1e-9);
            EXPECT_TRUE(query::frechet_within(db.trajectory(a), db.trajectory(b), expected));
            if (expected > 0.0) {
                EXPECT_FALSE(
                    query::frechet_within(db.trajectory(a), db.trajectory(b), expected * 0.999));
            }
            EXPECT_LE(query::endpoint_lower_bound(db.trajectory(a), db.trajectory(b)),
                      expected + 1e-9);
        }
    }
}

TEST(Join, MatchesBruteForce) {
    std::mt19937_64 rng(11);
    const auto db = make_db(random_tracks(rng, 200));
    const auto queries = make_db(random_tracks(rng, 20));
    for (const double eps : {100.0, 300.0, 800.0}) {
        for (std::size_t q = 0; q < queries.size(); ++q) {
            std::vector<std::size_t> expected;
            for (std::size_t i = 0; i < db.size(); ++i) {
                if (frechet(queries.trajectory(q), db.trajectory(i)) <= eps)
                    expected.push_back(i);
            }
            EXPECT_EQ(query::join(db, queries.trajectory(q), eps), expected)
                << "query " << q << ", eps " << eps;
        }
    }
}

TEST(Join, WorkloadStoresTheJoinOnTheOriginalData) {
    std::mt19937_64 rng(13);
    const auto db = make_db(random_tracks(rng, 150));
    const auto pool = make_db(random_tracks(rng, 30));
    const auto workload =
        query::generate_join_queries(db, pool, {.query_count = 0, .eps_m = 300.0});
    ASSERT_EQ(workload.entries.size(), pool.size());
    std::size_t pairs = 0;
    for (const auto &entry : workload.entries) {
        const auto query = workload.queries.trajectory(entry.query_index);
        EXPECT_EQ(entry.reference, query::join(db, query, workload.eps_m));
        EXPECT_TRUE(std::ranges::includes(entry.candidates, entry.reference));
        pairs += entry.reference.size();
    }
    EXPECT_GT(pairs, 0u);
}

TEST(Join, FixedEpsIsUsedAsGiven) {
    std::mt19937_64 rng(17);
    const auto db = make_db(random_tracks(rng, 50));
    const auto pool = make_db(random_tracks(rng, 10));
    const auto workload =
        query::generate_join_queries(db, pool, {.query_count = 4, .eps_m = 400.0});
    EXPECT_EQ(workload.entries.size(), 4u);
    EXPECT_DOUBLE_EQ(workload.eps_m, 400.0);
}

TEST(Join, RejectsInvalidConfig) {
    const auto db = make_db({{{39.9, 116.3}, {39.91, 116.3}}});
    EXPECT_THROW(query::generate_join_queries(db, db, {.eps_m = -1.0}), std::invalid_argument);
    EXPECT_THROW(query::generate_join_queries(db, db, {.eps_m = 0.0}), std::invalid_argument);
}

TEST(JoinEvaluator, UnsimplifiedDatabaseScoresPerfectly) {
    std::mt19937_64 rng(19);
    const auto db = make_db(random_tracks(rng, 120));
    const auto pool = make_db(random_tracks(rng, 25));
    const auto workload = query::generate_join_queries(db, pool, {.query_count = 0});
    const auto scores = evaluator::evaluate_join(db, workload);
    EXPECT_DOUBLE_EQ(scores.f1, 1.0);
    EXPECT_DOUBLE_EQ(scores.precision, 1.0);
    EXPECT_DOUBLE_EQ(scores.recall, 1.0);
    EXPECT_EQ(scores.query_count, pool.size());
}

TEST(JoinEvaluator, SimplifiedScoresMatchBruteForceJoin) {
    std::mt19937_64 rng(23);
    const auto db = make_db(random_tracks(rng, 150));
    const auto pool = make_db(random_tracks(rng, 20));
    const auto workload = query::generate_join_queries(db, pool, {.query_count = 0});
    const std::vector<std::size_t> budgets(db.size(), 3);
    const auto compressed = core::materialize(
        db, selection::simplify_database(db, selection::AlgorithmType::SED, budgets));

    // F1 per query against a join run on the whole simplified database, averaged by hand
    double f1_sum = 0.0;
    std::size_t counted = 0;
    for (const auto &entry : workload.entries) {
        const auto found =
            query::join(compressed, workload.queries.trajectory(entry.query_index), workload.eps_m);
        std::vector<std::size_t> hits;
        std::ranges::set_intersection(entry.reference, found, std::back_inserter(hits));
        if (found.empty() && entry.reference.empty())
            continue;
        f1_sum += 2.0 * static_cast<double>(hits.size()) /
                  static_cast<double>(found.size() + entry.reference.size());
        ++counted;
    }
    const auto scores = evaluator::evaluate_join(compressed, workload);
    ASSERT_GT(counted, 0u);
    EXPECT_NEAR(scores.f1, f1_sum / static_cast<double>(counted), 1e-12);
    EXPECT_LT(scores.f1, 1.0);
}

TEST(JoinEvaluator, EmptyWorkloadIsNaN) {
    const auto db = make_db({{{39.9, 116.3}, {39.91, 116.3}}});
    const query::JoinWorkload workload{core::subset(db, {}), {}, 100.0};
    const auto scores = evaluator::evaluate_join(db, workload);
    EXPECT_TRUE(std::isnan(scores.f1));
    EXPECT_EQ(scores.query_count, 0u);
}
