#include <gtest/gtest.h>

#include "backend/engine/core/projection.hpp"
#include "backend/engine/core/trajectory_db.hpp"
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
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

// Random walks in local metres, one point per second, starting near each other so that some
// pairs join
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
        return std::ranges::equal(x, y, [](const auto &p, const auto &q) {
            return p.i == q.i && p.j == q.j && p.seen == q.seen;
        });
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
    // A cache hit must reuse the file, rather than silently recomputing identical results.
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
    for (const std::uint64_t legacy : {0x5431'4341'4348'4532ull, 0x5431'4341'4348'4533ull}) {
        SCOPED_TRACE(legacy);
        {
            std::fstream out(file, std::ios::binary | std::ios::in | std::ios::out);
            out.write(reinterpret_cast<const char *>(&legacy), sizeof(legacy));
            ASSERT_TRUE(out);
        }
        // Even with matching settings and contents, an obsolete workload version is rejected.
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
    const teacher::Objective objective(db, workloads);

    const auto floor = endpoints(db);
    EXPECT_NEAR(objective.total(floor), 0.0, 1e-12);
    EXPECT_GT(objective.total(every_point(db)), 0.0);

    // Switch random trajectories between endpoints and every point; the tracked join metric
    // must equal a full recompute after each switch
    auto state = floor;
    const teacher::PreparedTrajectories prepared_floor(db, state);
    auto views = prepared_floor.views();
    teacher::JoinState join(objective, views);
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
        // Evaluating an alternative must neither change the state nor overwrite this proposal.
        const auto stay = join.propose(i, views[i], views);
        EXPECT_DOUBLE_EQ(stay.gain, 0.0);
        EXPECT_DOUBLE_EQ(join.metric(), old_metric);
        const double predicted = proposal.gain;
        const double separable = objective.separable(i, next) - objective.separable(i, state[i]);
        state[i] = next;
        views[i] = next_view;
        join.apply(proposal);
        EXPECT_NEAR(join.metric(), objective.metrics(state)[teacher::Objective::Join], 1e-12);
        EXPECT_NEAR(objective.total(state) - before, predicted + separable, 1e-9);
    }
}

TEST_F(TeacherTest, ShapeTermIsZeroAtTheFloorAndSeparable) {
    const auto db = random_db(6, 24, 30);
    const auto workloads = teacher::load_or_build_workloads(db, small_config(), 17, m_dir);
    const teacher::Objective plain(db, workloads);
    const teacher::Objective shaped(db, workloads, {.shape_weight = 0.25});
    const auto floor = endpoints(db), full = every_point(db);
    EXPECT_NEAR(shaped.total(floor), 0.0, 1e-12);
    // Keeping every point is a perfect shape, so the term adds exactly its weight
    EXPECT_NEAR(shaped.total(full) - plain.total(full), 0.25, 1e-9);
    for (std::size_t i = 0; i < db.size(); ++i) {
        EXPECT_DOUBLE_EQ(shaped.shape(i, db.trajectory(i)), 1.0);
        auto state = floor;
        state[i] = full[i];
        const double change = shaped.total(state) - shaped.total(floor);
        const double join = plain.total(state) - plain.total(floor) -
                            (plain.separable(i, full[i]) - plain.separable(i, floor[i]));
        EXPECT_NEAR(change, shaped.separable(i, full[i]) - shaped.separable(i, floor[i]) + join,
                    1e-9);
    }
}

TEST_F(TeacherTest, ShapeKnnTermNeverCreditsMoreThanTheQueryTerm) {
    const auto db = random_db(5, 24, 30);
    const auto workloads = teacher::load_or_build_workloads(db, small_config(), 15, m_dir);
    const teacher::Objective by_shape(db, workloads, {.knn_by_shape = true});
    const teacher::Objective by_query(db, workloads);
    // The shape condition implies the query one, so it can only credit fewer hits
    std::mt19937_64 rng(6);
    for (int trial = 0; trial < 20; ++trial) {
        teacher::KeptSets kept(db.size());
        for (std::size_t i = 0; i < db.size(); ++i) {
            const std::size_t n = db.trajectory(i).size();
            kept[i] = {0, n - 1};
            for (std::size_t k = 1; k + 1 < n; ++k)
                if (rng() % 4 == 0)
                    kept[i].push_back(k);
            std::ranges::sort(kept[i]);
        }
        EXPECT_LE(by_shape.metrics(kept)[teacher::Objective::Knn],
                  by_query.metrics(kept)[teacher::Objective::Knn] + 1e-12);
    }
    EXPECT_GT(by_shape.metrics(every_point(db))[teacher::Objective::Knn], 0.0);
}

TEST_F(TeacherTest, PreparedIndependentScoresMatchFullScoringAndSurvivePruning) {
    const auto db = random_db(2, 24, 20);
    const auto workloads = teacher::load_or_build_workloads(db, small_config(), 9, m_dir);
    std::mt19937_64 rng(74);
    for (const bool knn_by_shape : {false, true}) {
        for (const double shape_weight : {0.0, 0.25}) {
            const teacher::Objective objective(
                db, workloads, {.knn_by_shape = knn_by_shape, .shape_weight = shape_weight});
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
                            const bool matched = objective.match(views[i], other);
                            found += matched;
                            hits += matched && link.positive;
                        }
                        const double f1 = found + objective.truth(i)
                                              ? 2.0 * static_cast<double>(hits) /
                                                    static_cast<double>(found + objective.truth(i))
                                              : 0.0;
                        const double join = with_truth ? objective.scale(teacher::Objective::Join) *
                                                             f1 / static_cast<double>(with_truth)
                                                       : 0.0;
                        const double expected = objective.separable(i, views[i]) + join;
                        const auto score = objective.prepare_independent(i, views[i], partners);
                        EXPECT_GE(score.bound + 1e-12, expected);
                        EXPECT_DOUBLE_EQ(objective.independent_value(score, expected - 1e-6),
                                         expected);
                        EXPECT_LE(objective.independent_value(
                                      score, std::numeric_limits<double>::infinity()),
                                  expected + 1e-12);
                        EXPECT_DOUBLE_EQ(objective.independent_value(score), expected);
                        EXPECT_DOUBLE_EQ(objective.independent_value(i, views[i], partners),
                                         expected);
                    }
                }
            }
        }
    }
}

TEST_F(TeacherTest, ArmsKeepTheBudgetAndBestResponseOnlyImproves) {
    const auto db = random_db(3, 30, 25);
    auto workloads = teacher::load_or_build_workloads(db, small_config(), 11, m_dir);
    const teacher::Teacher model(db, std::move(workloads), {}, 11);
    const std::vector<std::string> arms(teacher::kArms.begin(), teacher::kArms.end());
    const std::size_t budget = 2 * db.size() + 40;
    const auto result = model.run(budget, arms);

    ASSERT_EQ(result.size(), arms.size());
    double gate_jt = 0, br_a0_jt = 0, teacher_jt = 0, br_jt = 0, iter_jt = 0;
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
        if (arm.name == "GATE_a0")
            gate_jt = arm.objective;
        if (arm.name == "BR_a0")
            br_a0_jt = arm.objective;
        if (arm.name == "TEACHER")
            teacher_jt = arm.objective;
        if (arm.name == "BR_TEACHER")
            br_jt = arm.objective;
        if (arm.name == "ITER_TEACHER")
            iter_jt = arm.objective;
    }
    EXPECT_GE(br_a0_jt, gate_jt - 1e-12);
    EXPECT_GE(br_jt, teacher_jt - 1e-12);
    EXPECT_GE(iter_jt, br_jt - 1e-12);
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
            if (b > 2 && b < n)
                EXPECT_TRUE(std::ranges::binary_search(kept, n / 2));
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

TEST(Teacher, RejectsUnknownArms) {
    EXPECT_TRUE(teacher::is_arm("GATE_a0"));
    EXPECT_FALSE(teacher::is_arm("gate_a0"));
    EXPECT_TRUE(teacher::is_arm("TEACHER"));
    EXPECT_FALSE(teacher::is_arm("TEACHER_A0"));
}

TEST_F(TeacherTest, IndividualArmsMatchCombinedRun) {
    const auto db = random_db(3, 30, 25);
    for (const auto variant :
         {query::FrechetVariant::Discrete, query::FrechetVariant::Continuous}) {
        auto config = small_config();
        config.frechet = variant;
        auto workloads = teacher::load_or_build_workloads(db, config, 11, m_dir);
        const teacher::Teacher model(db, std::move(workloads), {}, 11);
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
