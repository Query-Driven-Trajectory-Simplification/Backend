#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/query/knn.hpp"

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

// Random walks starting anywhere in a ~20 km area, so some are near each other and most are not
std::vector<Track> random_tracks(std::mt19937_64 &rng, std::size_t count) {
    std::uniform_real_distribution<double> start_lat(39.8, 40.0);
    std::uniform_real_distribution<double> start_lon(116.2, 116.45);
    std::uniform_int_distribution<std::size_t> length(1, 60);
    std::normal_distribution<double> step(0.0, 0.002);
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

// Full Fréchet distance to every trajectory, no pruning
std::vector<std::size_t> brute_force_knn(const core::TrajectoryDb &db,
                                         const core::TrajectoryView &query, std::size_t k) {
    std::vector<std::pair<double, std::size_t>> distances;
    for (std::size_t i = 0; i < db.size(); ++i) {
        distances.emplace_back(frechet(query, db.trajectory(i)), i);
    }
    std::ranges::sort(distances);
    std::vector<std::size_t> result;
    for (std::size_t i = 0; i < std::min(k, distances.size()); ++i) {
        result.push_back(distances[i].second);
    }
    return result;
}

} // namespace

TEST(Knn, FindsClosestTrajectoriesClosestFirst) {
    const auto db = make_db({
        {{39.90, 116.30}, {39.91, 116.30}}, // far
        {{39.950, 116.40}, {39.951, 116.40}},
        {{39.950, 116.40}, {39.952, 116.40}}, // exact match
        {{39.953, 116.40}, {39.954, 116.40}},
    });
    const auto query = db.trajectory(2);
    const std::vector<std::size_t> expected = {2, 1, 3};
    EXPECT_EQ(query::knn(db, query, 3), expected);
}

TEST(Knn, ZeroKReturnsNothing) {
    const auto db = make_db({{{39.9, 116.3}, {39.91, 116.3}}});
    EXPECT_TRUE(query::knn(db, db.trajectory(0), 0).empty());
}

TEST(Knn, KLargerThanDatabaseReturnsEverything) {
    std::mt19937_64 rng(3);
    const auto db = make_db(random_tracks(rng, 4));
    const auto query = db.trajectory(1);
    EXPECT_EQ(query::knn(db, query, 10), brute_force_knn(db, query, 10));
}

TEST(Knn, MatchesBruteForceOnRandomTrajectories) {
    std::mt19937_64 rng(7);
    const auto db = make_db(random_tracks(rng, 300));
    const auto queries = make_db(random_tracks(rng, 40));
    for (const std::size_t k : {1u, 3u, 5u, 10u}) {
        for (std::size_t q = 0; q < queries.size(); ++q) {
            const auto query = queries.trajectory(q);
            EXPECT_EQ(query::knn(db, query, k), brute_force_knn(db, query, k))
                << "query " << q << ", k " << k;
        }
    }
}
