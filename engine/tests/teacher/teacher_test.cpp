#include <gtest/gtest.h>

#include "backend/engine/core/projection.hpp"
#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/evaluator/range.hpp"
#include "backend/engine/teacher/greedy.hpp"
#include "backend/engine/teacher/objective.hpp"
#include "backend/engine/teacher/teacher.hpp"
#include "backend/engine/teacher/workloads.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

// Nearby random walks in local metres, one point per second, so that some pairs join
core::TrajectoryDb random_db(std::uint64_t seed, std::size_t count, std::size_t length) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> start(-200.0, 200.0);
    std::normal_distribution<double> step(40.0, 30.0);
    std::vector<std::uint64_t> offsets{0};
    std::vector<std::int32_t> ids;
    std::vector<std::int64_t> t;
    std::vector<double> lat, lon;
    for (std::size_t i = 0; i < count; ++i) {
        ids.push_back(static_cast<std::int32_t>(i));
        double x = start(rng), y = start(rng);
        for (std::size_t k = 0; k < length; ++k) {
            const auto ll = core::to_lat_lon(x, y);
            t.push_back(static_cast<std::int64_t>(k));
            lat.push_back(ll.lat);
            lon.push_back(ll.lon);
            x += step(rng);
            y += 0.5 * step(rng);
        }
        offsets.push_back(t.size());
    }
    return {std::move(offsets), std::move(ids), std::move(t), std::move(lat), std::move(lon)};
}

teacher::WorkloadConfig small_config() {
    teacher::WorkloadConfig config;
    config.k = 3;
    config.join_eps_m = 150.0;
    config.range = {.query_count = 60,
                    .box_size_m = 100.0,
                    .window_seconds = 604800,
                    .min_results = 1,
                    .full_accept_at = 1,
                    .attempts_per_query = 1000};
    config.where_at = {.probes_per_trajectory = 8, .max_gap_seconds = 60.0};
    return config;
}

class TeacherTest : public ::testing::Test {
  protected:
    void SetUp() override {
        m_dir = std::filesystem::temp_directory_path() /
                ("teacher_test_" + std::to_string(std::random_device{}()));
        std::filesystem::create_directories(m_dir);
    }
    void TearDown() override { std::filesystem::remove_all(m_dir); }
    std::filesystem::path m_dir;
};

teacher::KeptSets endpoints(const core::TrajectoryDb &db) {
    teacher::KeptSets kept(db.size());
    for (std::size_t i = 0; i < db.size(); ++i)
        kept[i] = {0, db.trajectory(i).size() - 1};
    return kept;
}

teacher::KeptSets every_point(const core::TrajectoryDb &db) {
    teacher::KeptSets kept(db.size());
    for (std::size_t i = 0; i < db.size(); ++i) {
        kept[i].resize(db.trajectory(i).size());
        for (std::size_t k = 0; k < kept[i].size(); ++k)
            kept[i][k] = k;
    }
    return kept;
}

std::string read_cache(const std::filesystem::path &file) {
    std::ifstream in(file, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void expect_same_cached_workloads(const teacher::TeacherWorkloads &a,
                                  const teacher::TeacherWorkloads &b) {
    ASSERT_EQ(a.knn.size(), b.knn.size());
    for (std::size_t i = 0; i < a.knn.size(); ++i) {
        EXPECT_EQ(a.knn[i].valid, b.knn[i].valid);
        EXPECT_EQ(a.knn[i].top, b.knn[i].top);
        EXPECT_DOUBLE_EQ(a.knn[i].cutoff, b.knn[i].cutoff);
    }
    auto same_pairs = [](const auto &x, const auto &y) {
        return std::ranges::equal(
            x, y, [](const auto &p, const auto &q) { return p.i == q.i && p.j == q.j; });
    };
    EXPECT_TRUE(same_pairs(a.positives, b.positives));
    EXPECT_TRUE(same_pairs(a.negatives, b.negatives));
}

} // namespace

TEST_F(TeacherTest, CacheIsWrittenThenReadBack) {
    const auto db = random_db(1, 24, 20);
    const auto built = teacher::load_or_build_workloads(db, small_config(), 5, m_dir);
    const auto file = m_dir / "teacher_cache_seed5.bin";
    ASSERT_TRUE(std::filesystem::exists(file));
    // A hit must reuse the file, not recompute
    const auto marker = std::filesystem::file_time_type::clock::now() - std::chrono::hours(24);
    std::filesystem::last_write_time(file, marker);
    const auto before = std::filesystem::last_write_time(file);
    const auto loaded = teacher::load_or_build_workloads(db, small_config(), 5, m_dir);

    EXPECT_EQ(std::filesystem::last_write_time(file), before);
    expect_same_cached_workloads(built, loaded);
    EXPECT_FALSE(built.positives.empty());
}

TEST_F(TeacherTest, CacheForChangedInteriorPointsIsRebuilt) {
    const auto source = random_db(1, 24, 20);
    auto make_db = [&](std::string_view change) {
        std::vector<std::uint64_t> offsets{0};
        std::vector<std::int32_t> ids;
        std::vector<std::int64_t> t;
        std::vector<double> lat, lon;
        for (std::size_t i = 0; i < source.size(); ++i) {
            const auto v = source.trajectory(i);
            ids.push_back(v.trajectory_id);
            for (std::size_t k = 0; k < v.size(); ++k) {
                t.push_back(2 * v.t[k]);
                lat.push_back(v.lat[k]);
                lon.push_back(v.lon[k]);
                if (i == 0 && k == v.size() / 2) {
                    if (change == "latitude")
                        lat.back() += 0.01;
                    if (change == "longitude")
                        lon.back() += 0.01;
                    if (change == "timestamp")
                        ++t.back();
                }
            }
            offsets.push_back(t.size());
        }
        return core::TrajectoryDb(std::move(offsets), std::move(ids), std::move(t), std::move(lat),
                                  std::move(lon));
    };
    const auto original = make_db("");
    for (const auto change : {"latitude", "longitude", "timestamp"}) {
        SCOPED_TRACE(change);
        const auto dir = m_dir / change;
        teacher::load_or_build_workloads(original, small_config(), 5, dir);
        const auto file = dir / "teacher_cache_seed5.bin";
        const auto before = read_cache(file);
        ASSERT_FALSE(before.empty());

        // IDs, lengths, endpoints and endpoint timestamps all stay unchanged.
        const auto changed = make_db(change);
        const auto rebuilt = teacher::load_or_build_workloads(changed, small_config(), 5, dir);
        EXPECT_NE(read_cache(file), before);
        const auto fresh =
            teacher::load_or_build_workloads(changed, small_config(), 5, dir / "fresh");
        expect_same_cached_workloads(rebuilt, fresh);
    }
}

TEST_F(TeacherTest, CacheWithLegacyFingerprintIsRebuilt) {
    const auto db = random_db(1, 24, 20);
    const auto built = teacher::load_or_build_workloads(db, small_config(), 5, m_dir);
    const auto file = m_dir / "teacher_cache_seed5.bin";
    const auto current = read_cache(file);
    ASSERT_FALSE(current.empty());
    for (const std::uint64_t legacy :
         {0x5431'4341'4348'4532ull, 0x5431'4341'4348'4533ull, 0x5431'4341'4348'4535ull}) {
        SCOPED_TRACE(legacy);
        {
            std::fstream out(file, std::ios::binary | std::ios::in | std::ios::out);
            out.write(reinterpret_cast<const char *>(&legacy), sizeof(legacy));
            ASSERT_TRUE(out);
        }
        // Reject obsolete fingerprints and V5's adaptive continuous-kNN reference ordering
        ASSERT_NE(read_cache(file), current);
        const auto rebuilt = teacher::load_or_build_workloads(db, small_config(), 5, m_dir);
        EXPECT_EQ(read_cache(file), current);
        expect_same_cached_workloads(built, rebuilt);
    }
}

TEST_F(TeacherTest, CacheForOtherSettingsIsRebuilt) {
    const auto db = random_db(1, 24, 20);
    teacher::load_or_build_workloads(db, small_config(), 5, m_dir);
    auto other = small_config();
    other.k = 2;
    const auto rebuilt = teacher::load_or_build_workloads(db, other, 5, m_dir);
    EXPECT_EQ(rebuilt.k, 2u);
    for (const auto &q : rebuilt.knn) {
        if (q.valid) {
            EXPECT_EQ(q.top.size(), 2u);
        }
    }
}

TEST_F(TeacherTest, ObjectiveIsZeroAtTheFloorAndTracksJoinExactly) {
    const auto db = random_db(2, 24, 20);
    const auto workloads = teacher::load_or_build_workloads(db, small_config(), 9, m_dir);
    for (const auto mode : {query::JoinMode::Self, query::JoinMode::HeldOut}) {
        SCOPED_TRACE(mode == query::JoinMode::Self ? "Self" : "HeldOut");
        const teacher::Objective objective(db, workloads, {.join_mode = mode});

        const auto floor = endpoints(db);
        EXPECT_NEAR(objective.total(floor), 0.0, 1e-12);
        EXPECT_GT(objective.total(every_point(db)), 0.0);

        // Switch random trajectories between endpoints and every point; the tracked join metric
        // must equal a full recompute after each switch
        auto state = floor;
        const teacher::PreparedTrajectories prepared_floor(db, state);
        auto views = prepared_floor.views();
        teacher::JoinState join(objective, views);
        teacher::RangeState range(objective, views);
        const auto full = every_point(db);
        const teacher::PreparedTrajectories prepared_full(db, full);
        const auto full_views = prepared_full.views();
        const auto floor_views = prepared_floor.views();
        std::mt19937_64 rng(4);
        for (int step = 0; step < 40; ++step) {
            const std::size_t i = rng() % db.size();
            const auto &next = state[i].size() == 2 ? full[i] : floor[i];
            const double before = objective.total(state);
            const auto next_view = state[i].size() == 2 ? full_views[i] : floor_views[i];
            const double old_metric = join.metric();
            const auto proposal = join.propose(i, next_view, views);
            const auto range_proposal = range.propose(i, next_view);
            // Proposing must not change the state or this proposal
            const auto stay = join.propose(i, views[i], views);
            EXPECT_DOUBLE_EQ(stay.gain, 0.0);
            EXPECT_DOUBLE_EQ(join.metric(), old_metric);
            const double predicted = proposal.gain;
            const double separable =
                objective.separable(i, next) - objective.separable(i, state[i]);
            state[i] = next;
            views[i] = next_view;
            join.apply(proposal);
            range.apply(range_proposal);
            EXPECT_NEAR(join.metric(), objective.metrics(state)[teacher::Objective::Join], 1e-12);
            EXPECT_NEAR(objective.total(state) - before,
                        predicted + range_proposal.gain + separable, 1e-9);
        }
    }
}

TEST_F(TeacherTest, RangeObjectiveUsesMacroF1EvenWhenRecallPrefersAnotherSolution) {
    std::vector<std::uint64_t> offsets{0};
    std::vector<std::int32_t> ids;
    std::vector<std::int64_t> times;
    std::vector<double> lat, lon;
    for (std::size_t i = 0; i < 8; ++i) {
        ids.push_back(static_cast<std::int32_t>(i));
        for (std::size_t k = 0; k < 3; ++k) {
            const auto point =
                core::to_lat_lon(k == 1 ? (i < 4 ? -100.0 : 100.0) : 0.0, k == 1 ? 0.0 : -100.0);
            times.push_back(static_cast<std::int64_t>(k));
            lat.push_back(point.lat);
            lon.push_back(point.lon);
        }
        offsets.push_back(times.size());
    }
    const core::TrajectoryDb db(offsets, ids, times, lat, lon);
    auto config = small_config();
    config.k = 1;
    config.join_eps_m = 50.0;
    auto workloads = teacher::load_or_build_workloads(db, config, 9, m_dir);
    workloads.range.queries = {
        {{-101, -99, -1, 1, 1, 1}, {0, 1, 2, 3}},
        {{99, 101, -1, 1, 1, 1}, {4, 5, 6, 7}},
        {{999, 1001, -1, 1, 1, 1}, {}}}; // Empty references do not enter the macro mean
    const teacher::Objective objective(db, workloads, {.term_weights = {0, 1, 0, 0}});
    auto concentrated = endpoints(db), balanced = endpoints(db);
    for (const auto i : {0u, 1u, 2u, 3u})
        concentrated[i] = {0, 1, 2};
    for (const auto i : {0u, 1u, 4u})
        balanced[i] = {0, 1, 2};
    const auto a =
        evaluator::evaluate_range(core::materialize(db, concentrated), workloads.range.queries);
    const auto b =
        evaluator::evaluate_range(core::materialize(db, balanced), workloads.range.queries);
    EXPECT_GT(a.recall, b.recall);
    EXPECT_LT(a.f1, b.f1);
    EXPECT_NEAR(objective.total(concentrated), a.f1, 1e-12);
    EXPECT_NEAR(objective.total(balanced), b.f1, 1e-12);
}

TEST_F(TeacherTest, RangeStateAndIndependentMarginalsMatchTheEvaluator) {
    const auto db = random_db(2, 24, 20);
    const auto workloads = teacher::load_or_build_workloads(db, small_config(), 9, m_dir);
    const teacher::Objective objective(db, workloads, {.term_weights = {0, 1, 0, 0}});
    auto state = endpoints(db);
    const auto floor = state, full = every_point(db);
    const teacher::PreparedTrajectories prepared_floor(db, floor), prepared_full(db, full);
    const auto floor_views = prepared_floor.views(), full_views = prepared_full.views();
    auto views = floor_views;
    teacher::RangeState range(objective, views);
    std::mt19937_64 rng(81);
    for (int step = 0; step < 60; ++step) {
        const auto i = rng() % db.size();
        const bool grow = state[i].size() == 2;
        const auto candidate = grow ? full_views[i] : floor_views[i];
        const auto before =
            evaluator::evaluate_range(core::materialize(db, state), workloads.range.queries).f1;
        const auto proposal = range.propose(i, candidate);
        EXPECT_DOUBLE_EQ(range.propose(i, views[i]).gain, 0.0);
        EXPECT_NEAR(range.metric(), before, 1e-12); // propose does not mutate

        // Independent range scores are the gain over i missing every box, with partners fixed.
        auto without_i = state;
        without_i[i].clear();
        const double without =
            evaluator::evaluate_range(core::materialize(db, without_i), workloads.range.queries).f1;
        const auto weights = range.weights(i);
        EXPECT_EQ(weights.size(), objective.boxes(i).size());
        const double independent = objective.independent_value(
            i, candidate, views, -std::numeric_limits<double>::infinity(), weights);
        state[i] = grow ? full[i] : floor[i];
        views[i] = candidate;
        range.apply(proposal);
        const double after =
            evaluator::evaluate_range(core::materialize(db, state), workloads.range.queries).f1;
        EXPECT_NEAR(range.metric(), after, 1e-12);
        EXPECT_NEAR(objective.metrics(state)[teacher::Objective::Range], after, 1e-12);
        EXPECT_NEAR(proposal.gain, objective.scale(teacher::Objective::Range) * (after - before),
                    1e-12);
        EXPECT_NEAR(independent, objective.scale(teacher::Objective::Range) * (after - without),
                    1e-12);
    }
    const teacher::RangeState original(objective, {});
    for (std::size_t i = 0; i < db.size(); ++i) {
        const auto weights = original.weights(i);
        const auto fallback = objective.range_weights(i, {});
        ASSERT_EQ(weights.size(), fallback.size());
        for (std::size_t k = 0; k < weights.size(); ++k) {
            EXPECT_DOUBLE_EQ(weights[k], fallback[k]);
            EXPECT_DOUBLE_EQ(weights[k], objective.boxes(i)[k].weight);
        }
    }
}

TEST_F(TeacherTest, SeveralJoinTermsAreTrackedExactly) {
    const auto db = random_db(2, 24, 20);
    auto config = small_config();
    config.extra_join_eps_m = {config.join_eps_m * 0.5, config.join_eps_m * 2.0};
    const auto workloads = teacher::load_or_build_workloads(db, config, 9, m_dir);
    ASSERT_EQ(workloads.extra_joins.size(), 2u);
    EXPECT_LE(workloads.extra_joins[0].positives.size(), workloads.positives.size());
    EXPECT_GE(workloads.extra_joins[1].positives.size(), workloads.positives.size());

    const teacher::Objective single(db, workloads, {.join_mode = query::JoinMode::Self});
    for (const bool held_out_term : {false, true}) {
        SCOPED_TRACE(held_out_term ? "with held-out term" : "self terms");
        const teacher::Objective objective(
            db, workloads, {.join_mode = query::JoinMode::Self, .held_out_term = held_out_term});
        EXPECT_EQ(objective.slots(), db.size() * (held_out_term ? 4u : 3u));
        EXPECT_NEAR(objective.total(endpoints(db)), 0.0, 1e-12);
        EXPECT_DOUBLE_EQ(objective.metrics(every_point(db))[teacher::Objective::Join], 1.0);

        auto state = endpoints(db);
        const auto floor = state, full = every_point(db);
        const teacher::PreparedTrajectories prepared_floor(db, floor), prepared_full(db, full);
        const auto floor_views = prepared_floor.views(), full_views = prepared_full.views();
        auto views = floor_views;
        teacher::JoinState join(objective, views);
        teacher::RangeState range(objective, views);
        std::mt19937_64 rng(4);
        for (int step = 0; step < 40; ++step) {
            const std::size_t i = rng() % db.size();
            const bool grow = state[i].size() == 2;
            const auto &next = grow ? full[i] : floor[i];
            const auto next_view = grow ? full_views[i] : floor_views[i];
            const double before = objective.total(state);
            const auto proposal = join.propose(i, next_view, views);
            const auto range_proposal = range.propose(i, next_view);
            const double separable =
                objective.separable(i, next) - objective.separable(i, state[i]);
            state[i] = next;
            views[i] = next_view;
            join.apply(proposal);
            range.apply(range_proposal);
            EXPECT_NEAR(join.metric(), objective.metrics(state)[teacher::Objective::Join], 1e-12);
            EXPECT_NEAR(objective.total(state) - before,
                        proposal.gain + range_proposal.gain + separable, 1e-9);
        }
    }
    // Without extra terms the workloads' extra joins are the only difference
    teacher::TeacherWorkloads plain = workloads;
    plain.extra_joins.clear();
    const teacher::Objective base(db, plain, {.join_mode = query::JoinMode::Self});
    EXPECT_EQ(base.slots(), db.size());
    EXPECT_GT(single.pair_count(), base.pair_count());
}

TEST_F(TeacherTest, JoinTermsAreNormalisedApartAndAveragedLikeJ4) {
    const auto db = random_db(2, 24, 20);
    auto config = small_config();
    config.extra_join_eps_m = {config.join_eps_m * 1.5, config.join_eps_m * 2.0};
    const auto workloads = teacher::load_or_build_workloads(db, config, 9, m_dir);
    ASSERT_FALSE(workloads.positives.empty());
    for (const auto &extra : workloads.extra_joins)
        ASSERT_FALSE(extra.positives.empty());
    const teacher::Objective objective(db, workloads, {.join_mode = query::JoinMode::Self});
    ASSERT_EQ(objective.join_terms(), 3u);

    auto macro_f1 = [&](const teacher::KeptSets &kept, std::size_t term) {
        const teacher::PreparedTrajectories prepared(db, kept);
        const auto views = prepared.views();
        double sum = 0;
        std::size_t counted = 0;
        for (std::size_t i = 0; i < db.size(); ++i) {
            std::size_t hits = 0, found = 0;
            for (const auto &link : objective.links(i)) {
                if (link.term != term)
                    continue;
                const bool matched = objective.match(views[i], views[link.other], term);
                found += matched;
                hits += matched && link.positive;
            }
            const auto truth = objective.truth(objective.slot(i, term));
            if (found + truth == 0)
                continue;
            sum += 2.0 * static_cast<double>(hits) / static_cast<double>(found + truth);
            ++counted;
        }
        return counted ? sum / static_cast<double>(counted) : 0.0;
    };

    std::mt19937_64 rng(31);
    auto kept = endpoints(db);
    for (std::size_t i = 0; i < db.size(); ++i) {
        for (std::size_t k = 1; k + 1 < db.trajectory(i).size(); ++k)
            if (rng() % 3 == 0)
                kept[i].push_back(k);
        std::ranges::sort(kept[i]);
    }
    double expected = 0;
    for (std::size_t term = 0; term < 3; ++term) {
        const double floor = macro_f1(endpoints(db), term);
        EXPECT_NEAR(objective.join_floor(term), floor, 1e-12);
        expected += (macro_f1(kept, term) - floor) / (1.0 - floor) / 3.0;
    }
    EXPECT_NEAR(objective.metrics(kept)[teacher::Objective::Join], expected, 1e-12);
}

TEST_F(TeacherTest, RankingKnnIsTrackedExactly) {
    const auto db = random_db(2, 24, 20);
    auto config = small_config();
    // Discrete distances order exactly, without the continuous ranking's 1 mm ties
    config.frechet = query::FrechetVariant::Discrete;
    const auto workloads = teacher::load_or_build_workloads(db, config, 9, m_dir);
    const teacher::Objective objective(db, workloads,
                                       {.join_mode = query::JoinMode::Self, .ranking_knn = true});
    const auto floor = endpoints(db), full = every_point(db);
    EXPECT_NEAR(objective.total(floor), 0.0, 1e-12);
    EXPECT_DOUBLE_EQ(objective.metrics(full)[teacher::Objective::Knn], 1.0);

    // Every query watches every trajectory, so the tracked value must equal a full recompute
    auto state = floor;
    const std::vector<teacher::KeptSets> watch{full};
    teacher::KnnState knn(objective, state, watch, db.size());
    EXPECT_NEAR(knn.metric(), objective.metrics(state)[teacher::Objective::Knn], 1e-12);
    const teacher::PreparedTrajectories prepared_floor(db, floor), prepared_full(db, full);
    const auto floor_views = prepared_floor.views(), full_views = prepared_full.views();
    auto views = floor_views;
    teacher::JoinState join(objective, views);
    teacher::RangeState range(objective, views);
    std::mt19937_64 rng(5);
    for (int step = 0; step < 40; ++step) {
        const std::size_t i = rng() % db.size();
        const bool grow = state[i].size() == 2;
        const auto &next = grow ? full[i] : floor[i];
        const auto next_view = grow ? full_views[i] : floor_views[i];
        const double before = objective.total(state);
        const auto distances = knn.distances(i, next_view);
        const double knn_gain = knn.propose(i, distances);
        EXPECT_DOUBLE_EQ(knn.propose(i, knn.distances(i, views[i])), 0.0);
        const auto proposal = join.propose(i, next_view, views);
        const auto range_proposal = range.propose(i, next_view);
        const double separable = objective.separable(i, next) - objective.separable(i, state[i]);
        state[i] = next;
        views[i] = next_view;
        join.apply(proposal);
        range.apply(range_proposal);
        knn.apply(i, distances);
        EXPECT_NEAR(knn.metric(), objective.metrics(state)[teacher::Objective::Knn], 1e-12);
        EXPECT_NEAR(objective.total(state) - before,
                    knn_gain + proposal.gain + range_proposal.gain + separable, 1e-9);
    }
}

TEST_F(TeacherTest, CompressedKnnQueriesAreTrackedExactly) {
    const auto db = random_db(2, 24, 20);
    auto config = small_config();
    config.frechet = query::FrechetVariant::Discrete;
    const auto workloads = teacher::load_or_build_workloads(db, config, 9, m_dir);
    const teacher::Objective objective(
        db, workloads,
        {.join_mode = query::JoinMode::Self, .ranking_knn = true, .compressed_knn_query = true});
    const auto floor = endpoints(db), full = every_point(db);
    EXPECT_NEAR(objective.total(floor), 0.0, 1e-12);
    EXPECT_DOUBLE_EQ(objective.metrics(full)[teacher::Objective::Knn], 1.0);

    // A switch changes i as a neighbour and as a query, both against the others' current state
    auto state = floor;
    const std::vector<teacher::KeptSets> watch{full};
    teacher::KnnState knn(objective, state, watch, db.size());
    EXPECT_NEAR(knn.metric(), objective.metrics(state)[teacher::Objective::Knn], 1e-12);
    const teacher::PreparedTrajectories prepared_floor(db, floor), prepared_full(db, full);
    const auto floor_views = prepared_floor.views(), full_views = prepared_full.views();
    auto views = floor_views;
    teacher::JoinState join(objective, views);
    teacher::RangeState range(objective, views);
    std::mt19937_64 rng(5);
    for (int step = 0; step < 40; ++step) {
        const std::size_t i = rng() % db.size();
        const bool grow = state[i].size() == 2;
        const auto &next = grow ? full[i] : floor[i];
        const auto next_view = grow ? full_views[i] : floor_views[i];
        const double before = objective.total(state);
        const auto move = knn.move(i, next_view, views);
        const double knn_gain = knn.propose(move);
        EXPECT_DOUBLE_EQ(knn.propose(knn.move(i, views[i], views)), 0.0);
        const auto proposal = join.propose(i, next_view, views);
        const auto range_proposal = range.propose(i, next_view);
        const double separable = objective.separable(i, next) - objective.separable(i, state[i]);
        state[i] = next;
        views[i] = next_view;
        join.apply(proposal);
        range.apply(range_proposal);
        knn.apply(move);
        EXPECT_NEAR(knn.metric(), objective.metrics(state)[teacher::Objective::Knn], 1e-12);
        EXPECT_NEAR(objective.total(state) - before,
                    knn_gain + proposal.gain + range_proposal.gain + separable, 1e-9);
    }
}

TEST_F(TeacherTest, CompressedKnnQueriesNeedTheRankingTerm) {
    const auto db = random_db(2, 24, 20);
    const auto workloads = teacher::load_or_build_workloads(db, small_config(), 9, m_dir);
    EXPECT_THROW(
        teacher::Objective(db, workloads, {.ranking_knn = false, .compressed_knn_query = true}),
        std::invalid_argument);
}

TEST_F(TeacherTest, ContinuousKnnNearTiesMatchFullScoringThroughSwitches) {
    // The first trajectory is a line; the next two backtrack by almost the same amount.
    // The old evaluator tied overlapping distance intervals, while the tracker sorted their
    // independently approximated distances: even the full database scored 23/24 instead of 1.
    const auto source = random_db(2, 24, 20);
    const std::array line{0.0, 3.0, 7.0, 10.0}, loop{0.0, 8.0, 2.0, 10.0};
    std::vector<std::uint64_t> offsets{0};
    std::vector<std::int32_t> ids;
    std::vector<std::int64_t> times;
    std::vector<double> lat, lon;
    for (std::size_t i = 0; i < source.size(); ++i) {
        ids.push_back(static_cast<std::int32_t>(i));
        const auto original = source.trajectory(i);
        for (std::size_t k = 0; k < (i < 3 ? line.size() : original.size()); ++k) {
            const double x = i >= 3   ? original.x[k]
                             : i == 0 ? line[k]
                                      : loop[k] + (i == 1 && k == 1 ? 0.001 : 0.0);
            const auto point = core::to_lat_lon(x, i < 3 ? 0.0 : original.y[k]);
            times.push_back(static_cast<std::int64_t>(k));
            lat.push_back(point.lat);
            lon.push_back(point.lon);
        }
        offsets.push_back(times.size());
    }
    const core::TrajectoryDb db(offsets, ids, times, lat, lon);
    auto config = small_config();
    config.k = 1;
    config.join_eps_m = 2.0;
    const auto workloads = teacher::load_or_build_workloads(db, config, 99, m_dir);
    const auto full = every_point(db), floor = endpoints(db);
    const teacher::PreparedTrajectories prepared_full(db, full), prepared_floor(db, floor);
    const auto full_views = prepared_full.views(), floor_views = prepared_floor.views();
    for (const bool compressed : {false, true}) {
        SCOPED_TRACE(compressed ? "compressed queries" : "original queries");
        const teacher::Objective objective(db, workloads,
                                           {.join_mode = query::JoinMode::HeldOut,
                                            .ranking_knn = true,
                                            .compressed_knn_query = compressed});
        auto kept = full;
        auto views = full_views;
        teacher::KnnState knn(objective, kept, {}, db.size());
        EXPECT_DOUBLE_EQ(objective.knn_recall(core::materialize(db, kept)), 1.0);
        EXPECT_DOUBLE_EQ(knn.metric(), 1.0);
        for (const auto i : {1u, 2u, 0u, 5u, 1u, 0u, 2u, 5u}) {
            const bool shrink = kept[i].size() != 2;
            const auto candidate = shrink ? floor_views[i] : full_views[i];
            const double before = objective.knn_recall(core::materialize(db, kept));
            const auto move = knn.move(i, candidate, views);
            const double gain = knn.propose(move);
            if (!compressed)
                EXPECT_NEAR(knn.propose_bounded(i, candidate), gain, 1e-12);
            kept[i] = shrink ? floor[i] : full[i];
            views[i] = candidate;
            knn.apply(move);
            const double after = objective.knn_recall(core::materialize(db, kept));
            EXPECT_NEAR(knn.metric(), after, 1e-12);
            EXPECT_NEAR(gain, objective.scale(teacher::Objective::Knn) * (after - before), 1e-12);
        }
    }
}

TEST_F(TeacherTest, CompressedKnnArmsKeepTheBudgetAndLoopOnlyImproves) {
    const auto db = random_db(3, 30, 25);
    auto workloads = teacher::load_or_build_workloads(db, small_config(), 11, m_dir);
    const teacher::Teacher model(
        db, std::move(workloads),
        {.join_mode = query::JoinMode::Self, .ranking_knn = true, .compressed_knn_query = true},
        11);
    const std::size_t budget = 2 * db.size() + 40;
    const std::vector<std::string> arms{"EQUAL_CHOICE",  "EQUAL_CHOICE_RDP_PARTNERS",
                                        "EQUAL_LOOP",    "BUDGET_RDP",
                                        "BUDGET_CHOICE", "BUDGET_LOOP"};
    const auto result = model.run(budget, arms);
    for (const auto &arm : result) {
        SCOPED_TRACE(arm.name);
        EXPECT_EQ(arm.simplification.num_points(), budget);
    }
    // Watch lists hold every trajectory here, so the loop's tracked gains are exact
    EXPECT_GE(result[2].objective, result[1].objective - 1e-12);
    EXPECT_GE(result[5].objective, result[4].objective - 1e-12);
}

TEST_F(TeacherTest, RankingKnnArmsKeepTheBudget) {
    const auto db = random_db(3, 30, 25);
    auto workloads = teacher::load_or_build_workloads(db, small_config(), 11, m_dir);
    const teacher::Teacher model(db, std::move(workloads),
                                 {.join_mode = query::JoinMode::HeldOut, .ranking_knn = true}, 11);
    const std::size_t budget = 2 * db.size() + 40;
    const std::vector<std::string> arms{"EQUAL_CHOICE", "EQUAL_LOOP", "BUDGET_RDP", "BUDGET_LOOP"};
    for (const auto &arm : model.run(budget, arms)) {
        SCOPED_TRACE(arm.name);
        EXPECT_EQ(arm.simplification.num_points(), budget);
        for (std::size_t i = 0; i < db.size(); ++i) {
            const auto kept = arm.simplification.trajectory(i);
            EXPECT_EQ(kept.front(), 0u);
            EXPECT_EQ(kept.back(), db.trajectory(i).size() - 1);
        }
    }
}

TEST_F(TeacherTest, TermWeightsScaleTheTerms) {
    const auto db = random_db(2, 24, 20);
    const auto workloads = teacher::load_or_build_workloads(db, small_config(), 9, m_dir);
    const teacher::Objective equal(db, workloads);
    const teacher::Objective weighted(db, workloads, {.term_weights = {2.0, 1.0, 1.0, 0.0}});
    using T = teacher::Objective;
    // Weights are relative: kNN gets 2/4 instead of 1/4, join drops out
    EXPECT_NEAR(weighted.scale(T::Knn), 2.0 * equal.scale(T::Knn), 1e-12);
    EXPECT_NEAR(weighted.scale(T::Range), equal.scale(T::Range), 1e-12);
    EXPECT_DOUBLE_EQ(weighted.scale(T::Join), 0.0);
    // Every point kept is perfect on every term, so J_T is the weight share of the terms used
    EXPECT_NEAR(weighted.total(every_point(db)), (2.0 + 1.0 + 1.0) / 4.0, 1e-9);
    EXPECT_THROW(teacher::Objective(db, workloads, {.term_weights = {0.0, 0.0, 0.0, 0.0}}),
                 std::invalid_argument);
    EXPECT_THROW(teacher::Objective(db, workloads, {.term_weights = {-1.0, 1.0, 1.0, 1.0}}),
                 std::invalid_argument);
}

TEST_F(TeacherTest, ExtraJoinThresholdsMustBeDistinctAndPositive) {
    const auto db = random_db(2, 24, 20);
    for (const auto &extra :
         {std::vector<double>{0.0}, std::vector<double>{-5.0},
          std::vector<double>{small_config().join_eps_m}, std::vector<double>{40.0, 40.0}}) {
        auto config = small_config();
        config.extra_join_eps_m = extra;
        EXPECT_THROW(teacher::load_or_build_workloads(db, config, 9, m_dir), std::invalid_argument);
    }
}

TEST_F(TeacherTest, HeldOutJoinScoresOriginalQueriesAgainstCompressedPartners) {
    const auto db = random_db(2, 24, 20);
    const auto workloads = teacher::load_or_build_workloads(db, small_config(), 9, m_dir);
    const teacher::Objective objective(db, workloads, {.join_mode = query::JoinMode::HeldOut});
    // Every point kept is the original database: each query finds exactly its positives
    EXPECT_DOUBLE_EQ(objective.metrics(every_point(db))[teacher::Objective::Join], 1.0);

    std::mt19937_64 rng(21);
    auto kept = endpoints(db);
    for (std::size_t i = 0; i < db.size(); ++i) {
        for (std::size_t k = 1; k + 1 < db.trajectory(i).size(); ++k)
            if (rng() % 4 == 0)
                kept[i].push_back(k);
        std::ranges::sort(kept[i]);
    }
    const teacher::PreparedTrajectories prepared(db, kept);
    const auto views = prepared.views();
    double sum = 0;
    std::size_t counted = 0;
    for (std::size_t e = 0; e < db.size(); ++e) {
        std::size_t hits = 0, found = 0;
        for (const auto &link : objective.links(e)) {
            const bool matched = objective.match(db.trajectory(e), views[link.other]);
            found += matched;
            hits += matched && link.positive;
        }
        if (found + objective.truth(e) == 0)
            continue;
        sum += 2.0 * static_cast<double>(hits) / static_cast<double>(found + objective.truth(e));
        ++counted;
    }
    ASSERT_GT(counted, 0u);
    // One term, normalised by its own floor
    const double floor = objective.join_floor(0);
    EXPECT_NEAR(objective.metrics(kept)[teacher::Objective::Join],
                (sum / static_cast<double>(counted) - floor) / (1.0 - floor), 1e-12);
}

TEST_F(TeacherTest, PreparedIndependentScoresMatchFullScoringAndSurvivePruning) {
    const auto db = random_db(2, 24, 20);
    const auto workloads = teacher::load_or_build_workloads(db, small_config(), 9, m_dir);
    std::mt19937_64 rng(74);
    for (const bool held_out : {false, true}) {
        {
            const teacher::Objective objective(
                db, workloads,
                {.join_mode = held_out ? query::JoinMode::HeldOut : query::JoinMode::Self});
            std::size_t with_truth = 0;
            for (std::size_t i = 0; i < db.size(); ++i)
                with_truth += objective.truth(i) != 0;
            for (int trial = 0; trial < 4; ++trial) {
                auto kept = endpoints(db);
                if (trial == 1)
                    kept = every_point(db);
                else if (trial > 1)
                    for (std::size_t i = 0; i < db.size(); ++i) {
                        for (std::size_t k = 1; k + 1 < db.trajectory(i).size(); ++k)
                            if (rng() % 4 == 0)
                                kept[i].push_back(k);
                        std::ranges::sort(kept[i]);
                    }
                const teacher::PreparedTrajectories prepared(db, kept);
                const auto views = prepared.views();
                for (const bool original_partners : {false, true}) {
                    const std::span<const core::TrajectoryView> partners =
                        original_partners ? std::span<const core::TrajectoryView>{} : views;
                    for (std::size_t i = 0; i < db.size(); ++i) {
                        std::size_t hits = 0, found = 0;
                        for (const auto &link : objective.links(i)) {
                            const auto other =
                                original_partners ? db.trajectory(link.other) : views[link.other];
                            // Held out, i is the database side of an original partner's query
                            const bool matched =
                                held_out ? objective.match(db.trajectory(link.other), views[i])
                                         : objective.match(views[i], other);
                            found += matched;
                            hits += matched && link.positive;
                        }
                        const double f1 = found + objective.truth(i)
                                              ? 2.0 * static_cast<double>(hits) /
                                                    static_cast<double>(found + objective.truth(i))
                                              : 0.0;
                        const double join = with_truth
                                                ? objective.scale(teacher::Objective::Join) * f1 /
                                                      (static_cast<double>(with_truth) *
                                                       (1.0 - objective.join_floor(0)))
                                                : 0.0;
                        double range_gain = 0.0;
                        const auto weights = objective.range_weights(i, partners);
                        const auto boxes = objective.boxes(i);
                        for (std::size_t b = 0; b < boxes.size(); ++b)
                            if (query::intersects(views[i],
                                                  workloads.range.queries[boxes[b].box].box))
                                range_gain += weights[b];
                        const double expected =
                            objective.separable(i, views[i]) +
                            objective.scale(teacher::Objective::Range) * range_gain + join;
                        const auto score = objective.prepare_independent(i, views[i], partners);
                        EXPECT_GE(score.bound + 1e-12, expected);
                        EXPECT_NEAR(objective.independent_value(score, expected - 1e-6), expected,
                                    1e-12);
                        EXPECT_LE(objective.independent_value(
                                      score, std::numeric_limits<double>::infinity()),
                                  expected + 1e-12);
                        EXPECT_NEAR(objective.independent_value(score), expected, 1e-12);
                        EXPECT_NEAR(objective.independent_value(i, views[i], partners), expected,
                                    1e-12);
                    }
                }
            }
        }
    }
}

TEST_F(TeacherTest, ArmsKeepTheBudgetAndBestResponseOnlyImproves) {
    const auto db = random_db(3, 30, 25);
    for (const auto mode : {query::JoinMode::Self, query::JoinMode::HeldOut})
        for (const bool ranking : {false, true}) {
            SCOPED_TRACE(std::string(mode == query::JoinMode::Self ? "Self" : "HeldOut") +
                         (ranking ? ", ranking kNN" : ""));
            auto workloads = teacher::load_or_build_workloads(db, small_config(), 11, m_dir);
            const teacher::Teacher model(
                db, std::move(workloads),
                {.join_mode = mode,
                 .ranking_knn = ranking,
                 .compressed_knn_query = ranking && mode == query::JoinMode::Self},
                11);
            const std::vector<std::string> arms(teacher::kArms.begin(), teacher::kArms.end());
            const std::size_t budget = 2 * db.size() + 40;
            const auto result = model.run(budget, arms);

            ASSERT_EQ(result.size(), arms.size());
            double rdp_partners_jt = 0, equal_loop_jt = 0, budget_choice_jt = 0, budget_loop_jt = 0,
                   reallocate_jt = 0, trade_jt = 0;
            for (const auto &arm : result) {
                SCOPED_TRACE(arm.name);
                ASSERT_EQ(arm.simplification.size(), db.size());
                EXPECT_EQ(arm.simplification.num_points(), budget);
                for (std::size_t i = 0; i < db.size(); ++i) {
                    const auto kept = arm.simplification.trajectory(i);
                    ASSERT_GE(kept.size(), 2u);
                    EXPECT_EQ(kept.front(), 0u);
                    EXPECT_EQ(kept.back(), db.trajectory(i).size() - 1);
                    EXPECT_TRUE(std::ranges::is_sorted(kept));
                    EXPECT_EQ(std::ranges::adjacent_find(kept), kept.end());
                }
                if (arm.name == "EQUAL_CHOICE_RDP_PARTNERS")
                    rdp_partners_jt = arm.objective;
                if (arm.name == "EQUAL_LOOP")
                    equal_loop_jt = arm.objective;
                if (arm.name == "BUDGET_CHOICE")
                    budget_choice_jt = arm.objective;
                if (arm.name == "BUDGET_LOOP")
                    budget_loop_jt = arm.objective;
                if (arm.name == "BUDGET_LOOP_REALLOCATE")
                    reallocate_jt = arm.objective;
                if (arm.name == "BUDGET_LOOP_TRADE")
                    trade_jt = arm.objective;
            }
            EXPECT_GE(equal_loop_jt, rdp_partners_jt - 1e-12);
            EXPECT_GE(budget_loop_jt, budget_choice_jt - 1e-12);
            EXPECT_GE(reallocate_jt, budget_loop_jt - 1e-12);
            EXPECT_GE(trade_jt, budget_loop_jt - 1e-12);
        }
}

// Each trajectory of random_db twice, the copy shift metres north
core::TrajectoryDb near_duplicates(std::uint64_t seed, std::size_t count, std::size_t length,
                                   double shift = 1e-4) {
    const auto source = random_db(seed, count, length);
    std::vector<std::uint64_t> offsets{0};
    std::vector<std::int32_t> ids;
    std::vector<std::int64_t> t;
    std::vector<double> lat, lon;
    for (std::size_t copy = 0; copy < 2; ++copy)
        for (std::size_t i = 0; i < source.size(); ++i) {
            const auto trajectory = source.trajectory(i);
            ids.push_back(static_cast<std::int32_t>(ids.size()));
            for (std::size_t k = 0; k < trajectory.size(); ++k) {
                const auto ll = core::to_lat_lon(
                    trajectory.x[k], trajectory.y[k] + shift * static_cast<double>(copy));
                t.push_back(trajectory.t[k]);
                lat.push_back(ll.lat);
                lon.push_back(ll.lon);
            }
            offsets.push_back(t.size());
        }
    return {std::move(offsets), std::move(ids), std::move(t), std::move(lat), std::move(lon)};
}

TEST_F(TeacherTest, BoundedKnnProposalsMatchFullDistances) {
    const auto db = near_duplicates(5, 24, 20);
    auto config = small_config();
    config.extra_join_eps_m.clear();
    config.k = 2;
    const auto workloads = teacher::load_or_build_workloads(db, config, 3, m_dir);
    const teacher::Objective objective(db, workloads, {.ranking_knn = true});
    std::mt19937_64 rng(8);
    auto random_kept = [&](std::size_t i) {
        const std::size_t n = db.trajectory(i).size();
        std::vector<std::size_t> kept{0};
        for (std::size_t k = 1; k + 1 < n; ++k)
            if (rng() % 3 == 0)
                kept.push_back(k);
        kept.push_back(n - 1);
        return kept;
    };
    for (const bool start_full : {true, false}) {
        teacher::KeptSets state = start_full ? every_point(db) : teacher::KeptSets(db.size());
        if (!start_full)
            for (std::size_t i = 0; i < db.size(); ++i)
                state[i] = random_kept(i);
        const teacher::KnnState knn(objective, state, {}, db.size());
        std::size_t changed = 0;
        for (std::size_t i = 0; i < db.size(); ++i)
            for (int trial = 0; trial < 8; ++trial) {
                const teacher::PreparedTrajectory candidate(db.trajectory(i), random_kept(i));
                const double full = knn.propose(i, knn.distances(i, candidate.view()));
                EXPECT_DOUBLE_EQ(knn.propose_bounded(i, candidate.view()), full) << i;
                changed += full != 0.0;
            }
        EXPECT_GT(changed, 0u);
    }
}

TEST_F(TeacherTest, WatchingVersionsMakesKnnProposalsExact) {
    // Copies 20 m apart: a copy with all points sits next to its twin, inside queries' top k
    const auto db = near_duplicates(6, 24, 20, 20.0);
    auto config = small_config();
    config.extra_join_eps_m.clear();
    const auto workloads = teacher::load_or_build_workloads(db, config, 3, m_dir);
    const teacher::Objective objective(db, workloads, {.ranking_knn = true});
    std::mt19937_64 rng(9);
    auto random_kept = [&](std::size_t i, std::size_t every) {
        const std::size_t n = db.trajectory(i).size();
        std::vector<std::size_t> kept{0};
        for (std::size_t k = 1; k + 1 < n; ++k)
            if (rng() % every == 0)
                kept.push_back(k);
        kept.push_back(n - 1);
        return kept;
    };
    teacher::KeptSets state(db.size());
    for (std::size_t i = 0; i < db.size(); ++i)
        state[i] = random_kept(i, 1000);
    const teacher::PreparedTrajectories prepared(db, state);
    const auto views = prepared.views();
    const double before = objective.knn_recall(core::materialize(db, state));
    const double scale = objective.scale(teacher::Objective::Knn);
    std::size_t changed = 0;
    for (std::size_t i = 0; i < db.size(); ++i) {
        // One spare place for whoever replaces a leaver, so versions entering a top k are the only
        // ones unwatched until watch()
        teacher::KnnState knn(objective, state, {}, workloads.k + 1);
        const auto kept = random_kept(i, 1);
        const teacher::PreparedTrajectory version(db.trajectory(i), kept);
        std::vector<std::vector<core::TrajectoryView>> versions(db.size());
        versions[i] = {version.view()};
        knn.watch(versions, views);
        auto next = state;
        next[i] = kept;
        const double exact = scale * (objective.knn_recall(core::materialize(db, next)) - before);
        EXPECT_NEAR(knn.propose(i, knn.distances(i, version.view())), exact, 1e-12) << i;
        EXPECT_NEAR(knn.propose_bounded(i, version.view()), exact, 1e-12) << i;
        changed += exact != 0.0;
    }
    EXPECT_GT(changed, 0u);
}

TEST_F(TeacherTest, GreedyKeepsTheBudgetAndTheSeed) {
    const auto db = random_db(4, 20, 30);
    const auto workloads = teacher::load_or_build_workloads(db, small_config(), 13, m_dir);
    const teacher::Objective objective(db, workloads);
    for (std::size_t i = 0; i < db.size(); ++i) {
        const teacher::Greedy greedy(objective, i);
        const std::size_t n = db.trajectory(i).size();
        const std::vector<std::size_t> seed{n / 2};
        std::vector<std::size_t> fill(n);
        for (std::size_t k = 0; k < n; ++k)
            fill[k] = n - 1 - k;
        for (const std::size_t b : {2u, 3u, 6u, 12u, 40u}) {
            SCOPED_TRACE(std::to_string(i) + " at " + std::to_string(b));
            const auto kept = greedy.run(
                b, b > 2 ? std::span<const std::size_t>(seed) : std::span<const std::size_t>(),
                fill);
            ASSERT_EQ(kept.size(), std::min(b, n));
            EXPECT_EQ(kept.front(), 0u);
            EXPECT_EQ(kept.back(), n - 1);
            EXPECT_TRUE(std::ranges::is_sorted(kept));
            EXPECT_EQ(std::ranges::adjacent_find(kept), kept.end());
            if (b > 2 && b < n) {
                EXPECT_TRUE(std::ranges::binary_search(kept, n / 2));
            }
        }
    }
}

TEST_F(TeacherTest, ExtremeBudgetsKeepEndpointsOrEveryPointForEveryArm) {
    const auto db = random_db(3, 30, 25);
    auto workloads = teacher::load_or_build_workloads(db, small_config(), 11, m_dir);
    const teacher::Teacher model(db, std::move(workloads), {}, 11);
    const std::vector<std::string> arms(teacher::kArms.begin(), teacher::kArms.end());
    for (const std::size_t budget : {2 * db.size(), db.num_points()}) {
        SCOPED_TRACE(budget);
        const auto result = model.run(budget, arms);
        ASSERT_EQ(result.size(), arms.size());
        for (const auto &arm : result) {
            SCOPED_TRACE(arm.name);
            EXPECT_EQ(arm.simplification.num_points(), budget);
            for (std::size_t i = 0; i < db.size(); ++i) {
                const auto n = db.trajectory(i).size();
                std::vector<std::size_t> expected{0};
                if (budget == db.num_points())
                    for (std::size_t k = 1; k + 1 < n; ++k)
                        expected.push_back(k);
                expected.push_back(n - 1);
                EXPECT_TRUE(std::ranges::equal(arm.simplification.trajectory(i), expected));
            }
        }
    }
}

TEST_F(TeacherTest, SingleRecipesKeepTheBudget) {
    const auto db = random_db(3, 30, 25);
    auto workloads = teacher::load_or_build_workloads(db, small_config(), 11, m_dir);
    const teacher::Teacher model(db, std::move(workloads), {}, 11);
    const std::size_t budget = 2 * db.size() + 40;
    const auto recipes = model.single_recipes(budget);
    ASSERT_EQ(recipes.size(), 6u);
    EXPECT_EQ(recipes.front().name, "RECIPE_RDP");
    for (const auto &arm : recipes) {
        SCOPED_TRACE(arm.name);
        EXPECT_EQ(arm.simplification.num_points(), budget);
    }
}

TEST(Teacher, RejectsUnknownArms) {
    EXPECT_TRUE(teacher::is_arm("EQUAL_CHOICE_RDP_PARTNERS"));
    EXPECT_FALSE(teacher::is_arm("gate_a0"));
    EXPECT_TRUE(teacher::is_arm("BUDGET_CHOICE"));
    EXPECT_FALSE(teacher::is_arm("TEACHER_A0"));
}

TEST_F(TeacherTest, IndividualArmsMatchCombinedRun) {
    const auto db = random_db(3, 30, 25);
    for (const auto [variant, ranking] : {std::pair{query::FrechetVariant::Discrete, false},
                                          std::pair{query::FrechetVariant::Continuous, false},
                                          std::pair{query::FrechetVariant::Continuous, true}}) {
        SCOPED_TRACE(ranking ? "ranking kNN" : "per-trajectory kNN");
        auto config = small_config();
        config.frechet = variant;
        auto workloads = teacher::load_or_build_workloads(db, config, 11, m_dir);
        const teacher::Teacher model(db, std::move(workloads), {.ranking_knn = ranking}, 11);
        const std::vector<std::string> arms(teacher::kArms.begin(), teacher::kArms.end());
        const std::size_t budget = 2 * db.size() + 40;
        const auto combined = model.run(budget, arms);
        for (std::size_t a = 0; a < arms.size(); ++a) {
            SCOPED_TRACE(arms[a]);
            const std::vector<std::string> requested{arms[a]};
            const auto alone = model.run(budget, requested);
            ASSERT_EQ(alone.size(), 1u);
            EXPECT_DOUBLE_EQ(alone[0].objective, combined[a].objective);
            for (std::size_t i = 0; i < db.size(); ++i)
                EXPECT_TRUE(std::ranges::equal(alone[0].simplification.trajectory(i),
                                               combined[a].simplification.trajectory(i)));
        }
    }
}

TEST(PreparedTrajectory, OwnsCoordinatesAndRebuildsBoundsAfterCopyMoveAndAssign) {
    const auto db = random_db(4, 2, 30);
    const auto original = db.trajectory(0);
    const std::vector<std::size_t> kept{3, 8, 12};
    const teacher::PreparedTrajectory prepared(original, kept);
    auto copied = prepared;
    auto moved = std::move(copied);
    const auto view = moved.view();
    EXPECT_NE(view.x.data(), prepared.view().x.data());
    EXPECT_TRUE(view.lat.empty());
    EXPECT_TRUE(view.lon.empty());
    for (std::size_t k = 0; k < kept.size(); ++k) {
        EXPECT_EQ(view.t[k], original.t[kept[k]]);
        EXPECT_EQ(view.x[k], original.x[kept[k]]);
        EXPECT_EQ(view.y[k], original.y[kept[k]]);
    }
    EXPECT_EQ(view.bbox.x_min, std::ranges::min(view.x));
    EXPECT_EQ(view.bbox.x_max, std::ranges::max(view.x));
    EXPECT_EQ(view.bbox.y_min, std::ranges::min(view.y));
    EXPECT_EQ(view.bbox.y_max, std::ranges::max(view.y));

    const auto replacement = db.trajectory(1);
    for (const auto &indices :
         {std::vector<std::size_t>{1, 29}, std::vector<std::size_t>{0, 4, 8, 16, 29}}) {
        moved.assign(replacement, indices);
        const auto updated = moved.view();
        EXPECT_EQ(updated.trajectory_id, replacement.trajectory_id);
        ASSERT_EQ(updated.size(), indices.size());
        for (std::size_t k = 0; k < indices.size(); ++k) {
            EXPECT_EQ(updated.t[k], replacement.t[indices[k]]);
            EXPECT_EQ(updated.x[k], replacement.x[indices[k]]);
            EXPECT_EQ(updated.y[k], replacement.y[indices[k]]);
        }
        EXPECT_EQ(updated.bbox.x_min, std::ranges::min(updated.x));
        EXPECT_EQ(updated.bbox.x_max, std::ranges::max(updated.x));
        EXPECT_EQ(updated.bbox.y_min, std::ranges::min(updated.y));
        EXPECT_EQ(updated.bbox.y_max, std::ranges::max(updated.y));
    }
}
