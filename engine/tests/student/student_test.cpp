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
    std::vector<std::int64_t> t;
    std::vector<double> lat, lon;

    for (int i = 0; i < 12; ++i) {
        ids.push_back(i);
        const int n = 40 + i * 15;
        for (int k = 0; k < n; ++k) {
            t.push_back(1700000000LL + static_cast<std::int64_t>(k) * 5 + i * 100);
            const double frac = static_cast<double>(k) / static_cast<double>(n - 1);
            const double dlat = 0.03 * frac + 0.005 * std::sin(frac * 6.28 * (1 + (i % 3)));
            const double dlon = 0.04 * frac * frac + 0.004 * std::cos(frac * 4.5 + i * 0.2);
            lat.push_back(39.90 + dlat + (i % 4) * 0.0005);
            lon.push_back(116.30 + dlon + (i / 4) * 0.0005);
        }
        offsets.push_back(t.size());
    }
    return core::TrajectoryDb(std::move(offsets), std::move(ids), std::move(t), std::move(lat),
                              std::move(lon));
}

void check_valid_kept(const core::TrajectoryView &tr, const std::vector<std::size_t> &kept,
                      std::size_t expected_b) {
    ASSERT_EQ(kept.size(), expected_b);
    ASSERT_EQ(kept.front(), 0U);
    ASSERT_EQ(kept.back(), tr.size() - 1);
    for (std::size_t j = 1; j < kept.size(); ++j)
        EXPECT_GT(kept[j], kept[j - 1]);
}

} // namespace

TEST(StudentTest, CandidatePoolPreservesEndpointsAndOrder) {
    const auto db = make_synthetic_db();
    const student::CandidatePoolConfig cfg{5.0, 250.0, 1000.0};
    for (std::size_t i = 0; i < db.size(); ++i) {
        const auto tr = db.trajectory(i);
        const auto pool = student::build_candidate_pool(tr, cfg);
        ASSERT_GE(pool.size(), 2U);
        EXPECT_EQ(pool.front(), 0U);
        EXPECT_EQ(pool.back(), tr.size() - 1);
        for (std::size_t j = 1; j < pool.size(); ++j)
            EXPECT_GT(pool[j], pool[j - 1]);
    }
}

TEST(StudentTest, Tiny10FeaturesAreExactAndFinite) {
    const auto db = make_synthetic_db();
    const student::DbDensityIndex dens(db);
    const auto feats = student::extract_trajectory_features(db, dens);

    EXPECT_EQ(feats.num_trajs, db.size());
    EXPECT_EQ(feats.feature_names.size(), 10U);
    EXPECT_EQ(feats.values.size(), db.size() * 10U);
    for (float v : feats.values)
        EXPECT_TRUE(std::isfinite(v));
}

TEST(StudentTest, SubmodularSolverPreservesExactBudget) {
    const auto db = make_synthetic_db();
    const student::DbDensityIndex dens(db);
    const student::CandidatePoolConfig pcfg{};

    for (std::size_t i = 0; i < db.size(); ++i) {
        const auto tr = db.trajectory(i);
        const auto pool = student::build_candidate_pool(tr, pcfg);
        for (std::size_t b : {std::size_t{2}, std::size_t{3}, std::size_t{8}, tr.size() / 2,
                              tr.size()}) {
            const auto kept =
                student::submodular_simplify_on_pool(tr, pool, b, {0.4, 1.3, 2.2, 2}, dens);
            check_valid_kept(tr, kept, b);
        }
    }
}

TEST(StudentTest, PureCppModelSimplifiesExactBudget) {
    const auto db = make_synthetic_db();
    student::StudentModel m;
    m.s1_in_dim = 10;
    m.num_anchors = 4;
    m.s1_mean.assign(10, 0.0f);
    m.s1_std.assign(10, 1.0f);
    m.log_anchors = {std::log(2.0f), std::log(4.0f), std::log(8.0f), std::log(16.0f)};
    m.s1_layers.push_back({10, 4, false, std::vector<float>(40, 0.05f), {1.0f, 0.2f, 0.2f, 0.2f}, {}, {}});

    m.s2_in_dim = 13;
    m.num_presets = 2;
    m.s2_mean.assign(13, 0.0f);
    m.s2_std.assign(13, 1.0f);
    m.log_simplex = {std::log(0.4f), std::log(1.3f), std::log(2.2f),
                     std::log(1.0f), std::log(1.0f), std::log(1.0f)};
    m.s2_layers.push_back({13, 2, false, std::vector<float>(26, 0.01f), {0.0f, 0.0f}, {}, {}});

    const std::size_t total_budget = db.size() * 2 + 60;
    const auto res = student::simplify_database(db, total_budget, m, 2);
    ASSERT_EQ(res.kept.size(), db.size());
    std::size_t sum = 0;
    for (std::size_t i = 0; i < db.size(); ++i) {
        check_valid_kept(db.trajectory(i), res.kept[i], res.kept[i].size());
        sum += res.kept[i].size();
    }
    EXPECT_EQ(sum, total_budget);
}
