#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/student/candidate_pool.hpp"
#include "backend/engine/student/density_index.hpp"
#include "backend/engine/student/features.hpp"
#include "backend/engine/student/student.hpp"
#include "backend/engine/student/submodular_solver.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

core::TrajectoryDb make_synthetic_db() {
    std::vector<std::uint64_t> offsets{0};
    std::vector<std::int32_t> ids;
    std::vector<std::int64_t> timestamps;
    std::vector<double> lat, lon;

    for (int i = 0; i < 12; ++i) {
        ids.push_back(i);
        const int num_points = 40 + i * 15;
        for (int k = 0; k < num_points; ++k) {
            timestamps.push_back(1700000000LL + static_cast<std::int64_t>(k) * 5 + i * 100);
            const double frac = static_cast<double>(k) / static_cast<double>(num_points - 1);
            const double dlat = 0.03 * frac + 0.005 * std::sin(frac * 6.28 * (1 + (i % 3)));
            const double dlon = 0.04 * frac * frac + 0.004 * std::cos(frac * 4.5 + i * 0.2);
            lat.push_back(39.90 + dlat + (i % 4) * 0.0005);
            lon.push_back(116.30 + dlon + (i / 4) * 0.0005);
        }
        offsets.push_back(timestamps.size());
    }
    return core::TrajectoryDb(std::move(offsets), std::move(ids), std::move(timestamps),
                              std::move(lat), std::move(lon));
}

void check_valid_kept(const core::TrajectoryView &traj, const std::vector<std::size_t> &kept,
                      std::size_t expected_budget) {
    ASSERT_EQ(kept.size(), expected_budget);
    ASSERT_EQ(kept.front(), 0U);
    ASSERT_EQ(kept.back(), traj.size() - 1);
    for (std::size_t j = 1; j < kept.size(); ++j)
        EXPECT_GT(kept[j], kept[j - 1]);
}

} // namespace

TEST(StudentTest, CandidatePoolPreservesEndpointsAndOrder) {
    const auto db = make_synthetic_db();
    const student::CandidatePoolConfig pool_cfg{5.0, 250.0, 1000.0};
    for (std::size_t i = 0; i < db.size(); ++i) {
        const auto traj = db.trajectory(i);
        const auto pool = student::build_candidate_pool(traj, pool_cfg);
        ASSERT_GE(pool.size(), 2U);
        EXPECT_EQ(pool.front(), 0U);
        EXPECT_EQ(pool.back(), traj.size() - 1);
        for (std::size_t j = 1; j < pool.size(); ++j)
            EXPECT_GT(pool[j], pool[j - 1]);
    }
}

TEST(StudentTest, FeaturesAreExactAndFinite) {
    const auto db = make_synthetic_db();
    const student::DbDensityIndex density_idx(db);
    const auto features = student::extract_trajectory_features(db, density_idx);

    EXPECT_EQ(features.num_trajs, db.size());
    EXPECT_EQ(features.feature_names.size(), 13U);
    EXPECT_EQ(features.values.size(), db.size() * 13U);
    for (float val : features.values)
        EXPECT_TRUE(std::isfinite(val));
}

TEST(StudentTest, SubmodularSolverPreservesExactBudget) {
    const auto db = make_synthetic_db();
    const student::DbDensityIndex density_idx(db);
    const student::CandidatePoolConfig pool_cfg{};

    for (std::size_t i = 0; i < db.size(); ++i) {
        const auto traj = db.trajectory(i);
        const auto pool = student::build_candidate_pool(traj, pool_cfg);
        for (std::size_t budget : {std::size_t{2}, std::size_t{3}, std::size_t{8}, traj.size() / 2,
                                   traj.size()}) {
            const auto kept = student::submodular_simplify_on_pool(
                traj, pool, budget, {0.4, 1.3, 2.2, 2}, density_idx);
            check_valid_kept(traj, kept, budget);
        }
    }
}

TEST(StudentTest, PureCppModelSimplifiesExactBudget) {
    const auto db = make_synthetic_db();
    student::StudentModel model;
    model.s1_in_dim = 13;
    model.num_anchors = 4;
    model.s1_mean.assign(13, 0.0f);
    model.s1_std.assign(13, 1.0f);
    model.log_anchors = {std::log(2.0f), std::log(4.0f), std::log(8.0f), std::log(16.0f)};
    model.s1_layers.push_back(
        {13, 4, false, std::vector<float>(52, 0.05f), {1.0f, 0.2f, 0.2f, 0.2f}, {}, {}});

    model.s2_in_dim = 16;
    model.num_presets = 2;
    model.s2_mean.assign(16, 0.0f);
    model.s2_std.assign(16, 1.0f);
    model.log_simplex = {std::log(0.4f), std::log(1.3f), std::log(2.2f),
                         std::log(1.0f), std::log(1.0f), std::log(1.0f)};
    model.s2_layers.push_back({16, 2, false, std::vector<float>(32, 0.01f), {0.0f, 0.0f}, {}, {}});

    const std::size_t total_budget = db.size() * 2 + 60;
    const auto result = student::simplify_database(db, total_budget, model, 2);
    ASSERT_EQ(result.kept.size(), db.size());
    std::size_t total_kept = 0;
    for (std::size_t i = 0; i < db.size(); ++i) {
        check_valid_kept(db.trajectory(i), result.kept[i], result.kept[i].size());
        total_kept += result.kept[i].size();
    }
    EXPECT_EQ(total_kept, total_budget);
}
