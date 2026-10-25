#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/split/split.hpp"

namespace {

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

void expect_same_trajectory(const core::TrajectoryView &a, const core::TrajectoryView &b) {
    EXPECT_EQ(a.trajectory_id, b.trajectory_id);
    EXPECT_TRUE(std::ranges::equal(a.t, b.t));
    EXPECT_TRUE(std::ranges::equal(a.lat, b.lat));
    EXPECT_TRUE(std::ranges::equal(a.lon, b.lon));
    EXPECT_TRUE(std::ranges::equal(a.x, b.x));
    EXPECT_TRUE(std::ranges::equal(a.y, b.y));
}

} // namespace

TEST(Subset, KeepsChosenTrajectoriesInGivenOrder) {
    const auto db = make_db({{0, 1, 2}, {5, 6}, {10, 11, 12, 13}});
    const std::vector<std::size_t> indices{2, 0};

    const auto result = core::subset(db, indices);

    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result.num_points(), 7u);
    expect_same_trajectory(result.trajectory(0), db.trajectory(2));
    expect_same_trajectory(result.trajectory(1), db.trajectory(0));
}

TEST(Subset, EmptyIndicesGiveEmptyDb) {
    const auto db = make_db({{0, 1, 2}, {5, 6}});

    const auto result = core::subset(db, {});

    EXPECT_EQ(result.size(), 0u);
    EXPECT_EQ(result.num_points(), 0u);
}

TEST(Subset, SplitPartsTogetherHoldEveryTrajectoryOnce) {
    const auto db = make_db({{0, 1}, {2, 3, 4}, {5, 6}, {7, 8, 9, 10}, {11, 12}, {13, 14, 15}});
    const auto parts = split::split(
        db, {.type = split::SplitType::Random, .query_percentage = 0.5, .seed = 7});

    const auto database = core::subset(db, parts.database);
    const auto queries = core::subset(db, parts.queries);

    EXPECT_EQ(database.size() + queries.size(), db.size());
    EXPECT_EQ(database.num_points() + queries.num_points(), db.num_points());
    for (std::size_t i = 0; i < database.size(); ++i)
        expect_same_trajectory(database.trajectory(i), db.trajectory(parts.database[i]));
    for (std::size_t i = 0; i < queries.size(); ++i)
        expect_same_trajectory(queries.trajectory(i), db.trajectory(parts.queries[i]));
}

TEST(Subset, SupportsRepeatedIndicesAndOwnsCopiedData) {
    const auto result = [] {
        const auto source = make_db({{0, 1}, {5, 6, 7}});
        const std::vector<std::size_t> indices{1, 1, 0};
        return core::subset(source, indices);
    }();
    const auto expected = make_db({{0, 1}, {5, 6, 7}});
    ASSERT_EQ(result.size(), 3u);
    expect_same_trajectory(result.trajectory(0), expected.trajectory(1));
    expect_same_trajectory(result.trajectory(1), expected.trajectory(1));
    expect_same_trajectory(result.trajectory(2), expected.trajectory(0));
}

TEST(Subset, RejectsOutOfRangeIndices) {
    const auto db = make_db({{0, 1}});
    const std::vector<std::size_t> indices{1};
    EXPECT_THROW(core::subset(db, indices), std::out_of_range);
}
