#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include "backend/engine/core/simplification.hpp"
#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/evaluator/combined.hpp"
#include "backend/engine/evaluator/route_clustering.hpp"
#include "backend/engine/query/join.hpp"
#include "backend/engine/query/route_clustering.hpp"

namespace {

using Track = std::vector<std::pair<double, double>>; // (lat, lon) per point

core::TrajectoryDb make_db(const std::vector<Track> &tracks,
                           const std::vector<std::int32_t> &ids = {}) {
    std::vector<std::uint64_t> offsets{0};
    std::vector<std::int32_t> trajectory_ids;
    std::vector<std::int64_t> t;
    std::vector<double> lat;
    std::vector<double> lon;
    for (std::size_t i = 0; i < tracks.size(); ++i) {
        trajectory_ids.push_back(ids.empty() ? static_cast<std::int32_t>(i) : ids[i]);
        for (std::size_t k = 0; k < tracks[i].size(); ++k) {
            t.push_back(static_cast<std::int64_t>(k));
            lat.push_back(tracks[i][k].first);
            lon.push_back(tracks[i][k].second);
        }
        offsets.push_back(t.size());
    }
    return {std::move(offsets), std::move(trajectory_ids), std::move(t), std::move(lat),
            std::move(lon)};
}

// One short track per trajectory, start points spread along longitude so tie-breaks are distinct
core::TrajectoryDb spread_db(std::size_t n) {
    std::vector<Track> tracks;
    for (std::size_t i = 0; i < n; ++i) {
        const double lon = 116.30 + 0.01 * static_cast<double>(i);
        tracks.push_back({{39.90, lon}, {39.91, lon}});
    }
    return make_db(tracks);
}

std::set<std::set<std::int32_t>> id_sets(const core::TrajectoryDb &db,
                                         const query::RouteClusters &clusters) {
    std::set<std::set<std::int32_t>> result;
    for (const auto &members : clusters.members) {
        std::set<std::int32_t> ids;
        for (const auto i : members)
            ids.insert(db.trajectory(i).trajectory_id);
        result.insert(ids);
    }
    return result;
}

// Trajectories on a few shared routes with small offsets, so routes form clusters
std::vector<Track> route_tracks(std::mt19937_64 &rng, std::size_t routes, std::size_t per_route) {
    std::normal_distribution<double> jitter(0.0, 0.0003);
    std::vector<Track> tracks;
    for (std::size_t r = 0; r < routes; ++r) {
        for (std::size_t m = 0; m < per_route; ++m) {
            Track track;
            for (std::size_t k = 0; k < 20; ++k) {
                const double s = static_cast<double>(k) / 19.0;
                track.emplace_back(39.90 + 0.05 * static_cast<double>(r) + 0.02 * s + jitter(rng),
                                   116.30 + 0.03 * s + 0.01 * std::sin(6.0 * s) + jitter(rng));
            }
            tracks.push_back(track);
        }
    }
    return tracks;
}

} // namespace

TEST(RouteClustering, StarFormsOneClusterAndLeavesNoiseUnclustered) {
    const auto db = spread_db(7);
    const std::vector<query::TrajectoryPair> pairs{{0, 1}, {0, 2}, {0, 3}, {0, 4}};
    const auto clusters = query::cluster_routes(db, pairs, 5);
    ASSERT_EQ(clusters.members.size(), 1u);
    EXPECT_EQ(clusters.members[0].front(), 0u);
    EXPECT_EQ(std::set<std::size_t>(clusters.members[0].begin(), clusters.members[0].end()),
              (std::set<std::size_t>{0, 1, 2, 3, 4}));
    EXPECT_EQ(clusters.label[5], -1);
    EXPECT_EQ(clusters.label[6], -1);
    EXPECT_TRUE(query::cluster_routes(db, pairs, 6).members.empty());
}

TEST(RouteClustering, MembersMustNeighbourTheMedoidSoPathsDoNotChain) {
    const auto db = spread_db(5);
    const std::vector<query::TrajectoryPair> path{{0, 1}, {1, 2}, {2, 3}, {3, 4}};
    const auto clusters = query::cluster_routes(db, path, 3);
    for (const auto &members : clusters.members) {
        EXPECT_LE(members.size(), 3u);
        const auto medoid = members.front();
        for (std::size_t k = 1; k < members.size(); ++k) {
            const auto a = std::min(medoid, members[k]);
            const auto b = std::max(medoid, members[k]);
            EXPECT_NE(std::find(path.begin(), path.end(), query::TrajectoryPair{a, b}), path.end());
        }
    }
    // Ties on degree go to the smallest start x, which is trajectory 1
    ASSERT_EQ(clusters.members.size(), 1u);
    EXPECT_EQ(clusters.members[0].front(), 1u);
}

TEST(RouteClustering, ResultDoesNotDependOnTrajectoryOrder) {
    std::mt19937_64 rng(7);
    const auto tracks = route_tracks(rng, 4, 6);
    std::vector<std::int32_t> ids(tracks.size());
    std::iota(ids.begin(), ids.end(), 0);
    const auto db = make_db(tracks, ids);

    std::vector<std::size_t> order(tracks.size());
    std::iota(order.begin(), order.end(), 0);
    std::ranges::shuffle(order, rng);
    std::vector<Track> shuffled;
    std::vector<std::int32_t> shuffled_ids;
    for (const auto i : order) {
        shuffled.push_back(tracks[i]);
        shuffled_ids.push_back(ids[i]);
    }
    const auto permuted = make_db(shuffled, shuffled_ids);

    const auto a = query::generate_route_clustering(db, {.enabled = true, .min_size = 3});
    const auto b = query::generate_route_clustering(permuted, {.enabled = true, .min_size = 3});
    EXPECT_FALSE(a.reference.members.empty());
    EXPECT_EQ(id_sets(db, a.reference), id_sets(permuted, b.reference));
}

TEST(RouteClustering, ReusesAMatchingSelfJoin) {
    std::mt19937_64 rng(3);
    const auto db = make_db(route_tracks(rng, 3, 5));
    const auto join = query::generate_self_join(db, 500.0);
    const auto reused = query::generate_route_clustering(
        db, {.enabled = true, .eps_m = 500.0, .min_size = 3}, &join);
    const auto fresh =
        query::generate_route_clustering(db, {.enabled = true, .eps_m = 500.0, .min_size = 3});
    EXPECT_EQ(reused.candidates, fresh.candidates);
    EXPECT_EQ(reused.reference.members, fresh.reference.members);
}

TEST(RouteClustering, IgnoresASelfJoinOfAnotherDatabaseWithTheSameSize) {
    std::mt19937_64 rng(3);
    const auto db = make_db(route_tracks(rng, 3, 5));
    const auto other = make_db(std::vector<Track>(db.size(), {{39.0, 116.0}, {39.0, 116.001}}));
    const auto other_join = query::generate_self_join(other, 500.0);
    const query::RouteClusteringConfig config{.enabled = true, .eps_m = 500.0, .min_size = 3};
    const auto workload = query::generate_route_clustering(db, config, &other_join);
    const auto fresh = query::generate_route_clustering(db, config);
    EXPECT_EQ(workload.candidates, fresh.candidates);
    EXPECT_EQ(workload.reference.members, fresh.reference.members);
}

TEST(RouteClustering, RepeatedPairsCountOnce) {
    const auto db = spread_db(3);
    const std::vector<query::TrajectoryPair> repeated{{0, 1}, {1, 0}, {0, 1}};
    EXPECT_TRUE(query::cluster_routes(db, repeated, 3).members.empty());
    const std::vector<query::TrajectoryPair> star{{0, 1}, {1, 0}, {0, 2}};
    const auto clusters = query::cluster_routes(db, star, 3);
    ASSERT_EQ(clusters.members.size(), 1u);
    EXPECT_EQ(clusters.members[0].size(), 3u);
}

TEST(RouteClustering, UncompressedDatabaseScoresPerfectly) {
    std::mt19937_64 rng(11);
    const auto db = make_db(route_tracks(rng, 4, 6));
    const auto workload = query::generate_route_clustering(db, {.enabled = true, .min_size = 3});
    ASSERT_FALSE(workload.reference.members.empty());
    const auto scores = evaluator::evaluate_route_clustering(db, workload);
    EXPECT_DOUBLE_EQ(scores.score, 1.0);
    EXPECT_EQ(scores.predicted_clusters, scores.reference_clusters);
}

TEST(RouteClustering, CompressedDatabaseIsScoredAgainstTheReference) {
    std::mt19937_64 rng(5);
    const auto db = make_db(route_tracks(rng, 4, 6));
    const auto workload = query::generate_route_clustering(db, {.enabled = true, .min_size = 3});
    core::Simplification endpoints;
    for (std::size_t i = 0; i < db.size(); ++i) {
        const std::vector<std::size_t> kept{0, db.trajectory(i).size() - 1};
        endpoints.add_trajectory(kept);
    }
    const auto compressed = core::materialize(db, endpoints);
    const auto scores = evaluator::evaluate_route_clustering(compressed, workload);
    EXPECT_GE(scores.score, 0.0);
    EXPECT_LE(scores.score, 1.0);
    const auto direct = evaluator::compare_route_clusters(
        workload.reference, evaluator::compressed_route_clusters(compressed, workload));
    EXPECT_DOUBLE_EQ(scores.score, direct.score);
}

TEST(RouteClustering, CompareScoresOverlapFromBothSides) {
    query::RouteClusters reference{{0, 0, 0, 0, -1}, {{0, 1, 2, 3}}};
    query::RouteClusters predicted{{0, 0, -1, -1, -1}, {{0, 1}}};
    const auto scores = evaluator::compare_route_clusters(reference, predicted);
    // Both sides' best F1 is 2 * 2 / (4 + 2)
    EXPECT_NEAR(scores.score, 2.0 / 3.0, 1e-12);
    EXPECT_DOUBLE_EQ(evaluator::compare_route_clusters(reference, reference).score, 1.0);

    const query::RouteClusters none{{-1, -1, -1, -1, -1}, {}};
    EXPECT_DOUBLE_EQ(evaluator::compare_route_clusters(reference, none).score, 0.0);
    EXPECT_TRUE(std::isnan(evaluator::compare_route_clusters(none, predicted).score));
}

TEST(RouteClustering, RejectsMismatchedDatabasesAndInvalidInput) {
    std::mt19937_64 rng(2);
    const auto db = make_db(route_tracks(rng, 2, 4));
    const auto workload = query::generate_route_clustering(db, {.enabled = true, .min_size = 3});
    const auto other = make_db(route_tracks(rng, 2, 3));
    EXPECT_THROW(evaluator::evaluate_route_clustering(other, workload), std::invalid_argument);
    // Same IDs and order, but a moved endpoint could match outside the cached candidates
    std::vector<Track> endpoints;
    for (std::size_t i = 0; i < db.size(); ++i) {
        const auto t = db.trajectory(i);
        endpoints.push_back({{t.lat.front(), t.lon.front()}, {t.lat.back(), t.lon.back()}});
    }
    EXPECT_NO_THROW(evaluator::evaluate_route_clustering(make_db(endpoints), workload));
    endpoints[0].back().second += 0.01;
    EXPECT_THROW(evaluator::evaluate_route_clustering(make_db(endpoints), workload),
                 std::invalid_argument);
    EXPECT_THROW(query::cluster_routes(db, {}, 1), std::invalid_argument);
    const std::vector<query::TrajectoryPair> self_pair{{0, 0}};
    EXPECT_THROW(query::cluster_routes(db, self_pair, 3), std::invalid_argument);
    EXPECT_THROW(query::generate_route_clustering(db, {.enabled = true, .eps_m = -1.0}),
                 std::invalid_argument);
}

TEST(RouteClustering, CombinedIncludesClustering) {
    evaluator::Scoretable floor{};
    floor.knn.recall_at_k = 0.0;
    floor.range_f1 = 0.0;
    floor.where_at.average_hit = 0.0;
    floor.join.f1 = 0.0;
    floor.clustering.score = 0.5;
    auto scores = floor;
    scores.knn.recall_at_k = 1.0;
    scores.clustering.score = 0.75;
    const auto normalized = evaluator::normalize_scores(scores, floor);
    EXPECT_DOUBLE_EQ(normalized.clustering, 0.5);
    EXPECT_DOUBLE_EQ(normalized.combined, 1.5 / 5.0);
    scores.clustering.score = std::numeric_limits<double>::quiet_NaN();
    EXPECT_DOUBLE_EQ(evaluator::normalize_scores(scores, floor).combined, 0.25);
}
