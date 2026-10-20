#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "backend/engine/core/simplification.hpp"
#include "backend/engine/core/trajectory_db.hpp"

namespace {

using Indices = std::vector<std::size_t>;

// Builds a TrajectoryDb from per-trajectory timestamps. Ids are 10, 11, ... and every point
// gets its own coordinates, so copied points can be told apart
core::TrajectoryDb make_db(const std::vector<std::vector<std::int64_t>> &trajectories) {
    std::vector<std::uint64_t> offsets{0};
    std::vector<std::int32_t> ids;
    std::vector<std::int64_t> t;
    std::vector<double> lat;
    std::vector<double> lon;
    for (std::size_t i = 0; i < trajectories.size(); ++i) {
        ids.push_back(static_cast<std::int32_t>(10 + i));
        for (const std::int64_t time : trajectories[i]) {
            const auto k = static_cast<double>(t.size());
            t.push_back(time);
            lat.push_back(39.9 + 0.001 * k);
            lon.push_back(116.3 + 0.002 * k);
        }
        offsets.push_back(t.size());
    }
    return {std::move(offsets), std::move(ids), std::move(t), std::move(lat), std::move(lon)};
}

core::Simplification make_simplification(const std::vector<Indices> &kept) {
    core::Simplification simplification;
    for (const auto &indices : kept)
        simplification.add_trajectory(indices);
    return simplification;
}

} // namespace

TEST(Simplification, StoresKeptIndicesPerTrajectory) {
    const auto simplification = make_simplification({{0, 4}, {0, 1, 2}});

    ASSERT_EQ(simplification.size(), 2u);
    EXPECT_EQ(simplification.num_points(), 5u);
    EXPECT_TRUE(std::ranges::equal(simplification.trajectory(0), Indices{0, 4}));
    EXPECT_TRUE(std::ranges::equal(simplification.trajectory(1), Indices{0, 1, 2}));
}

TEST(Materialize, KeepsOnlyKeptPointsOfEveryTrajectory) {
    const auto db = make_db({{0, 1, 2, 3, 4}, {10, 11, 12}});
    const auto simplification = make_simplification({{0, 2, 4}, {0, 2}});

    const auto result = core::materialize(db, simplification);

    ASSERT_EQ(result.size(), db.size());
    EXPECT_EQ(result.num_points(), 5u);
    for (std::size_t i = 0; i < db.size(); ++i) {
        const auto original = db.trajectory(i);
        const auto compressed = result.trajectory(i);
        const auto kept = simplification.trajectory(i);
        EXPECT_EQ(compressed.trajectory_id, original.trajectory_id);
        ASSERT_EQ(compressed.size(), kept.size());
        for (std::size_t j = 0; j < kept.size(); ++j) {
            EXPECT_EQ(compressed.t[j], original.t[kept[j]]);
            EXPECT_EQ(compressed.lat[j], original.lat[kept[j]]);
            EXPECT_EQ(compressed.lon[j], original.lon[kept[j]]);
            EXPECT_EQ(compressed.x[j], original.x[kept[j]]);
            EXPECT_EQ(compressed.y[j], original.y[kept[j]]);
        }
    }
}

TEST(Materialize, KeepingEverythingReproducesDatabase) {
    const auto db = make_db({{0, 1, 2}, {5, 6}});
    const auto simplification = make_simplification({{0, 1, 2}, {0, 1}});

    const auto result = core::materialize(db, simplification);

    ASSERT_EQ(result.size(), db.size());
    for (std::size_t i = 0; i < db.size(); ++i) {
        EXPECT_TRUE(std::ranges::equal(result.trajectory(i).t, db.trajectory(i).t));
        EXPECT_TRUE(std::ranges::equal(result.trajectory(i).x, db.trajectory(i).x));
        EXPECT_TRUE(std::ranges::equal(result.trajectory(i).y, db.trajectory(i).y));
    }
}

TEST(Materialize, ThrowsWhenSizesDiffer) {
    const auto db = make_db({{0, 1, 2}, {5, 6}});
    const auto simplification = make_simplification({{0, 2}});

    EXPECT_THROW(core::materialize(db, simplification), std::invalid_argument);
}
