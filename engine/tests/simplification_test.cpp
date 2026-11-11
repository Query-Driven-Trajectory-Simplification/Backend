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

TEST(Materialize, RebuildsBoundsAfterRemovingExtrema) {
    const core::TrajectoryDb db({0, 4}, {10}, {0, 1, 2, 3}, {39.9, 39.8, 40.1, 40.0},
                                {116.3, 116.2, 116.6, 116.4});
    const auto result = core::materialize(db, make_simplification({{0, 3}}));
    const auto source = db.trajectory(0);
    const auto compressed = result.trajectory(0);
    EXPECT_EQ(compressed.bbox.x_min, std::min(source.x[0], source.x[3]));
    EXPECT_EQ(compressed.bbox.x_max, std::max(source.x[0], source.x[3]));
    EXPECT_EQ(compressed.bbox.y_min, std::min(source.y[0], source.y[3]));
    EXPECT_EQ(compressed.bbox.y_max, std::max(source.y[0], source.y[3]));
    EXPECT_GT(compressed.bbox.x_min, source.bbox.x_min);
    EXPECT_LT(compressed.bbox.x_max, source.bbox.x_max);
    EXPECT_GT(compressed.bbox.y_min, source.bbox.y_min);
    EXPECT_LT(compressed.bbox.y_max, source.bbox.y_max);
}

TEST(Materialize, PreservesEmptyAndSinglePointTrajectories) {
    const auto db = make_db({{}, {0, 1}, {2}});
    const auto result = core::materialize(db, make_simplification({{}, {}, {0}}));
    ASSERT_EQ(result.size(), 3u);
    EXPECT_EQ(result.num_points(), 1u);
    EXPECT_EQ(result.trajectory(0).size(), 0u);
    EXPECT_EQ(result.trajectory(1).size(), 0u);
    const auto empty_box = db.trajectory(0).bbox;
    EXPECT_EQ(result.trajectory(1).bbox.x_min, empty_box.x_min);
    EXPECT_EQ(result.trajectory(1).bbox.x_max, empty_box.x_max);
    EXPECT_EQ(result.trajectory(1).bbox.y_min, empty_box.y_min);
    EXPECT_EQ(result.trajectory(1).bbox.y_max, empty_box.y_max);
    const auto single = result.trajectory(2);
    EXPECT_EQ(single.bbox.x_min, single.x[0]);
    EXPECT_EQ(single.bbox.x_max, single.x[0]);
    EXPECT_EQ(single.bbox.y_min, single.y[0]);
    EXPECT_EQ(single.bbox.y_max, single.y[0]);
    const auto empty = core::materialize(make_db({}), make_simplification({}));
    EXPECT_EQ(empty.size(), 0u);
    EXPECT_EQ(empty.num_points(), 0u);
}

TEST(Materialize, RejectsInvalidKeptIndices) {
    const auto db = make_db({{0, 1, 2}});
    EXPECT_THROW(core::materialize(db, make_simplification({{0, 3}})), std::out_of_range);
    EXPECT_THROW(core::materialize(db, make_simplification({{0, 0}})), std::invalid_argument);
    EXPECT_THROW(core::materialize(db, make_simplification({{2, 1}})), std::invalid_argument);
}
