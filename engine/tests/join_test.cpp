#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "backend/engine/core/simplification.hpp"
#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/evaluator/join.hpp"
#include "backend/engine/query/frechet_helper.hpp"
#include "backend/engine/query/join.hpp"
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
            EXPECT_EQ(query::join(db, queries.trajectory(q), eps, query::FrechetVariant::Discrete),
                      expected)
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

TEST(JoinEvaluator, FilteredHeldOutWorkloadIsScoredPerEntry) {
    std::mt19937_64 rng(29);
    const auto db = make_db(random_tracks(rng, 60));
    const auto pool = make_db(random_tracks(rng, 6));
    auto workload = query::generate_join_queries(db, pool, {.query_count = 0});
    const std::vector<std::size_t> budgets(db.size(), 3);
    const auto compressed = core::materialize(
        db, selection::simplify_database(db, selection::AlgorithmType::SED, budgets));
    const auto full = evaluator::evaluate_join(compressed, workload);
    // Keep only the last query, whose index is past the size of the filtered workload
    workload.entries.erase(workload.entries.begin(), workload.entries.end() - 1);
    const auto kept = evaluator::evaluate_join(compressed, workload);
    EXPECT_EQ(kept.query_count, 1u);
    const auto &entry = workload.entries.front();
    const auto found =
        query::join(compressed, workload.queries.trajectory(entry.query_index), workload.eps_m);
    std::vector<std::size_t> hits;
    std::ranges::set_intersection(entry.reference, found, std::back_inserter(hits));
    if (!found.empty() || !entry.reference.empty())
        EXPECT_DOUBLE_EQ(kept.f1, 2.0 * static_cast<double>(hits.size()) /
                                      static_cast<double>(found.size() + entry.reference.size()));
    EXPECT_EQ(full.query_count, pool.size());
}

TEST(JoinEvaluator, EmptyWorkloadIsNaN) {
    const auto db = make_db({{{39.9, 116.3}, {39.91, 116.3}}});
    const query::JoinWorkload workload{core::subset(db, {}), {}, 100.0};
    const auto scores = evaluator::evaluate_join(db, workload);
    EXPECT_TRUE(std::isnan(scores.f1));
    EXPECT_EQ(scores.query_count, 0u);
}

TEST(SelfJoin, BothOperandsAreSimplifiedAndPairsAreUnordered) {
    const Track bend{{39.9, 116.3}, {39.92, 116.3}, {39.9, 116.31}};
    const auto db = make_db({bend, bend});
    const auto workload = query::generate_self_join(db, 500);
    ASSERT_EQ(workload.entries.size(), 2u);
    EXPECT_EQ(workload.entries[0].candidates, (std::vector<std::size_t>{1}));
    EXPECT_EQ(workload.entries[0].reference, (std::vector<std::size_t>{1}));
    EXPECT_TRUE(workload.entries[1].candidates.empty());
    core::Simplification one, both;
    const std::vector<std::size_t> endpoints{0, 2}, full{0, 1, 2};
    one.add_trajectory(endpoints);
    one.add_trajectory(full);
    both.add_trajectory(endpoints);
    both.add_trajectory(endpoints);
    EXPECT_DOUBLE_EQ(evaluator::evaluate_join(core::materialize(db, one), workload).f1, 0);
    const auto scores = evaluator::evaluate_join(core::materialize(db, both), workload);
    EXPECT_DOUBLE_EQ(scores.f1, 1);
    EXPECT_EQ(scores.reference_pairs, 1u);
    EXPECT_EQ(scores.predicted_pairs, 1u);
    EXPECT_EQ(scores.true_positive_pairs, 1u);
    EXPECT_EQ(scores.query_count, 2u);
    EXPECT_EQ(scores.mode, query::JoinMode::Self);
}

TEST(SelfJoin, CountsFalsePositivesPerTrajectory) {
    const Track up{{39.9, 116.3}, {39.92, 116.3}, {39.9, 116.31}};
    const Track down{{39.9, 116.3}, {39.88, 116.3}, {39.9, 116.31}};
    const auto db = make_db({up, up, down});
    const auto workload = query::generate_self_join(db);
    core::Simplification s;
    const std::vector<std::size_t> endpoints{0, 2};
    for (int i = 0; i < 3; ++i)
        s.add_trajectory(endpoints);
    const auto scores = evaluator::evaluate_join(core::materialize(db, s), workload);
    EXPECT_EQ(scores.candidate_pairs, 3u);
    EXPECT_EQ(scores.reference_pairs, 1u);
    EXPECT_EQ(scores.predicted_pairs, 3u);
    EXPECT_EQ(scores.true_positive_pairs, 1u);
    // Trajectories 0 and 1 find their true partner plus a false one, F1 2/3 each. Trajectory 2
    // has no true partner but finds two, F1 0.
    EXPECT_DOUBLE_EQ(scores.f1, 4.0 / 9);
    EXPECT_DOUBLE_EQ(scores.precision, 1.0 / 3);
    EXPECT_DOUBLE_EQ(scores.recall, 1);
    EXPECT_DOUBLE_EQ(scores.micro_f1, 0.5);
    EXPECT_DOUBLE_EQ(scores.micro_precision, 1.0 / 3);
    EXPECT_DOUBLE_EQ(scores.micro_recall, 1);
}

TEST(SelfJoin, EveryTrajectoryWeighsTheSameHoweverManyPartnersItHas) {
    // Four copies of a straight line keep all 6 of their pairs, while a bend loses its only
    // partner. Pooling pairs hides the loss; averaging over trajectories does not.
    const Track bend{{39.9, 116.3}, {39.92, 116.3}, {39.9, 116.31}};
    const Track line{{40.5, 116.3}, {40.5, 116.31}};
    const auto db = make_db({bend, bend, line, line, line, line});
    const auto workload = query::generate_self_join(db, 500);
    core::Simplification s;
    s.add_trajectory(std::vector<std::size_t>{0, 2});
    s.add_trajectory(std::vector<std::size_t>{0, 1, 2});
    for (int i = 0; i < 4; ++i)
        s.add_trajectory(std::vector<std::size_t>{0, 1});
    const auto scores = evaluator::evaluate_join(core::materialize(db, s), workload);
    EXPECT_EQ(scores.reference_pairs, 7u);
    EXPECT_EQ(scores.true_positive_pairs, 6u);
    EXPECT_DOUBLE_EQ(scores.micro_f1, 12.0 / 13);
    EXPECT_DOUBLE_EQ(scores.f1, 4.0 / 6);
    EXPECT_DOUBLE_EQ(scores.recall, 4.0 / 6);
    EXPECT_DOUBLE_EQ(scores.precision, 1);
}

TEST(SelfJoin, MatchesIndependentFullGridOnAllOriginalAndCompressedPairs) {
    std::mt19937_64 rng(478);
    const auto db = make_db(random_tracks(rng, 30));
    core::Simplification s;
    for (std::size_t i = 0; i < db.size(); ++i) {
        const auto n = db.trajectory(i).size();
        std::vector<std::size_t> kept{0};
        for (std::size_t k = 1; k + 1 < n; ++k)
            if (rng() % 3 == 0)
                kept.push_back(k);
        kept.push_back(n - 1);
        s.add_trajectory(kept);
    }
    const auto compressed = core::materialize(db, s);
    for (double eps : {100., 500., 1500.}) {
        const auto workload = query::generate_self_join(db, eps, query::FrechetVariant::Discrete);
        std::size_t truth = 0, predicted = 0, hits = 0;
        for (std::size_t i = 0; i < db.size(); ++i) {
            std::vector<std::size_t> expected;
            for (std::size_t j = i + 1; j < db.size(); ++j) {
                const bool ref = frechet(db.trajectory(i), db.trajectory(j)) <= eps;
                const bool pred =
                    frechet(compressed.trajectory(i), compressed.trajectory(j)) <= eps;
                if (ref)
                    expected.push_back(j);
                truth += ref;
                predicted += pred;
                hits += ref && pred;
            }
            EXPECT_EQ(workload.entries[i].reference, expected);
        }
        const auto scores = evaluator::evaluate_join(compressed, workload);
        EXPECT_EQ(scores.reference_pairs, truth);
        EXPECT_EQ(scores.predicted_pairs, predicted);
        EXPECT_EQ(scores.true_positive_pairs, hits);
        if (truth + predicted)
            EXPECT_DOUBLE_EQ(scores.micro_f1, 2.0 * hits / (truth + predicted));
        else
            EXPECT_TRUE(std::isnan(scores.micro_f1));
        // Every trajectory as a query over all others
        double f1_sum = 0.0;
        std::size_t counted = 0;
        for (std::size_t i = 0; i < db.size(); ++i) {
            std::size_t ref = 0, pred = 0, both = 0;
            for (std::size_t j = 0; j < db.size(); ++j) {
                if (j == i)
                    continue;
                const bool r = frechet(db.trajectory(i), db.trajectory(j)) <= eps;
                const bool p = frechet(compressed.trajectory(i), compressed.trajectory(j)) <= eps;
                ref += r;
                pred += p;
                both += r && p;
            }
            if (ref + pred) {
                f1_sum += 2.0 * static_cast<double>(both) / static_cast<double>(ref + pred);
                ++counted;
            }
        }
        if (counted)
            EXPECT_NEAR(scores.f1, f1_sum / static_cast<double>(counted), 1e-12);
        else
            EXPECT_TRUE(std::isnan(scores.f1));
        if (truth)
            EXPECT_DOUBLE_EQ(evaluator::evaluate_join(db, workload).f1, 1);
    }
}

TEST(SelfJoin, ConfigModeIgnoresHeldOutPoolSamplingAndSeed) {
    const auto db = make_db({{{39.9, 116.3}}, {{39.9, 116.3}}, {{39.9, 116.3}}});
    const query::JoinConfig config{.query_count = 1, .mode = query::JoinMode::Self};
    const auto workload = query::generate_join_queries(db, make_db({}), config, 912);
    const auto scores = evaluator::evaluate_join(db, workload);
    EXPECT_EQ(scores.reference_pairs, 3u);
    EXPECT_DOUBLE_EQ(scores.f1, 1);
}

TEST(SelfJoin, IncludesThresholdBoundaryAndPrunesEitherEndpoint) {
    const auto db = make_db({{{39.9, 116.3}, {39.9, 116.31}},
                             {{39.901, 116.3}, {39.901, 116.31}},
                             {{39.9, 116.3}, {40.0, 116.31}},
                             {{40.0, 116.3}, {39.9, 116.31}}});
    const double eps = query::endpoint_lower_bound(db.trajectory(0), db.trajectory(1));
    const auto workload = query::generate_self_join(db, eps);
    EXPECT_EQ(workload.entries[0].candidates, (std::vector<std::size_t>{1}));
    EXPECT_EQ(workload.entries[0].reference, (std::vector<std::size_t>{1}));
}

TEST(SelfJoin, EmptyPairSetsAreUndefinedAndInvalidInputsAreRejected) {
    for (const auto &db :
         {make_db({}), make_db({{{39.9, 116.3}}}), make_db({{{39.9, 116.3}}, {{40.9, 116.3}}})}) {
        const auto scores = evaluator::evaluate_join(db, query::generate_self_join(db));
        EXPECT_TRUE(std::isnan(scores.f1));
        EXPECT_TRUE(std::isnan(scores.precision));
        EXPECT_TRUE(std::isnan(scores.recall));
        EXPECT_TRUE(std::isnan(scores.micro_f1));
    }
    const auto db = make_db({{{39.9, 116.3}}, {{39.9, 116.3}}});
    for (double eps : {0., -1., std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()})
        EXPECT_THROW(query::generate_self_join(db, eps), std::invalid_argument);
    EXPECT_THROW(query::generate_self_join(make_db({{}})), std::invalid_argument);
    const auto workload = query::generate_self_join(db);
    EXPECT_THROW(evaluator::evaluate_join(make_db({}), workload), std::invalid_argument);
    EXPECT_THROW(evaluator::evaluate_join(make_db({{{40.0, 116.3}}, {{39.9, 116.3}}}), workload),
                 std::invalid_argument);
    const std::vector<std::size_t> reverse{1, 0};
    EXPECT_THROW(evaluator::evaluate_join(core::subset(db, reverse), workload),
                 std::invalid_argument);
}

TEST(SelfJoin, EmptyReferenceWithFalsePositivesScoresZero) {
    const auto db = make_db({{{39.9, 116.3}, {39.92, 116.3}, {39.9, 116.31}},
                             {{39.9, 116.3}, {39.88, 116.3}, {39.9, 116.31}}});
    core::Simplification s;
    const std::vector<std::size_t> endpoints{0, 2};
    s.add_trajectory(endpoints);
    s.add_trajectory(endpoints);
    const auto scores =
        evaluator::evaluate_join(core::materialize(db, s), query::generate_self_join(db));
    EXPECT_EQ(scores.reference_pairs, 0u);
    EXPECT_EQ(scores.predicted_pairs, 1u);
    EXPECT_DOUBLE_EQ(scores.f1, 0);
    EXPECT_DOUBLE_EQ(scores.precision, 0);
    EXPECT_TRUE(std::isnan(scores.recall));
}
