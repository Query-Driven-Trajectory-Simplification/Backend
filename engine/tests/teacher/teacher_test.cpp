#include <gtest/gtest.h>

#include "backend/engine/core/projection.hpp"
#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/teacher/objective.hpp"
#include "backend/engine/teacher/teacher.hpp"
#include "backend/engine/teacher/workloads.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
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

} // namespace

TEST_F(TeacherTest, CacheIsWrittenThenReadBack) {
    const auto db = random_db(1, 24, 20);
    const auto built = teacher::load_or_build_workloads(db, small_config(), 5, m_dir);
    ASSERT_TRUE(std::filesystem::exists(m_dir / "teacher_cache_seed5.bin"));
    const auto loaded = teacher::load_or_build_workloads(db, small_config(), 5, m_dir);

    ASSERT_EQ(built.knn.size(), loaded.knn.size());
    for (std::size_t e = 0; e < built.knn.size(); ++e) {
        EXPECT_EQ(built.knn[e].valid, loaded.knn[e].valid);
        EXPECT_EQ(built.knn[e].top, loaded.knn[e].top);
        EXPECT_DOUBLE_EQ(built.knn[e].cutoff, loaded.knn[e].cutoff);
    }
    auto same = [](const std::vector<teacher::JoinPair> &a, const std::vector<teacher::JoinPair> &b) {
        return std::ranges::equal(a, b, [](const auto &x, const auto &y) {
            return x.i == y.i && x.j == y.j && x.seen == y.seen;
        });
    };
    EXPECT_TRUE(same(built.positives, loaded.positives));
    EXPECT_TRUE(same(built.negatives, loaded.negatives));
    EXPECT_FALSE(built.positives.empty());
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
    teacher::JoinState join(objective, state);
    const auto full = every_point(db);
    std::mt19937_64 rng(4);
    for (int step = 0; step < 40; ++step) {
        const std::size_t i = rng() % db.size();
        const auto &next = state[i].size() == 2 ? full[i] : floor[i];
        const double before = objective.total(state);
        const double predicted = join.delta(i, next, state);
        const double separable = objective.separable(i, next) - objective.separable(i, state[i]);
        state[i] = next;
        join.apply(i, state);
        EXPECT_NEAR(join.metric(), objective.metrics(state)[teacher::Objective::Join], 1e-12);
        EXPECT_NEAR(objective.total(state) - before, predicted + separable, 1e-9);
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
    double teacher_jt = 0, br_jt = 0, gate_jt = 0, br_a0_jt = 0;
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
        if (arm.name == "TEACHER")
            teacher_jt = arm.objective;
        if (arm.name == "BR_TEACHER")
            br_jt = arm.objective;
        if (arm.name == "GATE_a0")
            gate_jt = arm.objective;
        if (arm.name == "BR_a0")
            br_a0_jt = arm.objective;
    }
    EXPECT_GE(br_jt, teacher_jt - 1e-12);
    EXPECT_GE(br_a0_jt, gate_jt - 1e-12);
}

TEST(Teacher, RejectsUnknownArms) {
    EXPECT_TRUE(teacher::is_arm("TEACHER"));
    EXPECT_FALSE(teacher::is_arm("teacher"));
}
