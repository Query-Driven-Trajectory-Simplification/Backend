#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/split/split.hpp"
#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

// Builds a TrajectoryDb where each trajectory i has 1 point at timestamp t[i]
core::TrajectoryDb make_db(std::vector<std::int64_t> t) {
    const auto n = t.size();
    std::vector<std::uint64_t> offsets(n + 1);
    std::iota(offsets.begin(), offsets.end(), 0);
    return {std::move(offsets), std::vector<std::int32_t>(n), std::move(t), std::vector<double>(n),
            std::vector<double>(n)};
}

} // namespace

TEST(RandomSplitTest, ExactSizesAndDisjointIndices) {
    auto db = make_db({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto result = split::split(db, split::SplitType::Random, 0.25, 42);

    // 1. 25% queries on 10 items -> 7 in database, 3 in queries
    EXPECT_EQ(result.database.size(), 7u);
    EXPECT_EQ(result.queries.size(), 3u);

    // 2. Selection sampling keeps both vectors sorted in ascending order
    EXPECT_TRUE(std::is_sorted(result.database.begin(), result.database.end()));
    EXPECT_TRUE(std::is_sorted(result.queries.begin(), result.queries.end()));

    // 3. Merging database + queries yields every index 0..9 exactly once
    std::vector<std::size_t> merged;
    std::merge(result.database.begin(), result.database.end(), result.queries.begin(),
               result.queries.end(), std::back_inserter(merged));
    EXPECT_EQ(merged, (std::vector<std::size_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
}

TEST(RandomSplitTest, SameSeedReconstructsExactSameSplit) {
    auto db = make_db({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

    auto run1 = split::split(db, split::SplitType::Random, 0.3, 12345);
    auto run2 = split::split(db, split::SplitType::Random, 0.3, 12345);

    EXPECT_EQ(run1.database, run2.database);
    EXPECT_EQ(run1.queries, run2.queries);
}

TEST(RandomSplitTest, DifferentSeedsProduceDifferentSplits) {
    std::vector<std::int64_t> times(50);
    std::iota(times.begin(), times.end(), 0);
    auto db = make_db(std::move(times));

    auto run1 = split::split(db, split::SplitType::Random, 0.3, 42);
    auto run2 = split::split(db, split::SplitType::Random, 0.3, 1337);

    EXPECT_NE(run1.queries, run2.queries);
    EXPECT_NE(run1.database, run2.database);
}

TEST(RandomSplitTest, ZeroPercentQueries) {
    auto db = make_db({10, 20, 30, 40});
    auto result = split::split(db, split::SplitType::Random, 0.0, 42);

    EXPECT_EQ(result.database, (std::vector<std::size_t>{0, 1, 2, 3}));
    EXPECT_TRUE(result.queries.empty());
}

TEST(RandomSplitTest, HundredPercentQueries) {
    auto db = make_db({10, 20, 30, 40});
    auto result = split::split(db, split::SplitType::Random, 1.0, 42);

    EXPECT_TRUE(result.database.empty());
    EXPECT_EQ(result.queries, (std::vector<std::size_t>{0, 1, 2, 3}));
}

TEST(RandomSplitTest, EmptyDatabase) {
    auto db = make_db({});
    auto result = split::split(db, split::SplitType::Random, 0.5, 42);

    EXPECT_TRUE(result.database.empty());
    EXPECT_TRUE(result.queries.empty());
}

TEST(RandomSplitTest, InvalidPercentageThrows) {
    auto db = make_db({1, 2, 3});

    EXPECT_THROW(split::split(db, split::SplitType::Random, -0.1, 42), std::invalid_argument);
    EXPECT_THROW(split::split(db, split::SplitType::Random, 1.1, 42), std::invalid_argument);
}