#include "backend/engine/split/split.hpp"
#include "backend/engine/core/trajectory_db.hpp"
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
    return {
        std::move(offsets),
        std::vector<std::int32_t>(n),
        std::move(t),
        std::vector<double>(n),
        std::vector<double>(n)
    };
}

} // namespace

TEST(TimeSplitTest, SortsUnsortedTrajectories) {
    auto db = make_db({40, 10, 30, 20});
    auto result = split::split(db, split::SplitType::Time, 0.5);

    EXPECT_EQ(result.database, (std::vector<std::size_t>{1, 3}));
    EXPECT_EQ(result.queries, (std::vector<std::size_t>{2, 0}));
}

// The time split rounds up the amout of query indices, as seen in this test below
TEST(TimeSplitTest, StandardSplit) {
    auto db = make_db({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto result = split::split(db, split::SplitType::Time, 0.25);

    EXPECT_EQ(result.database, (std::vector<std::size_t>{0, 1, 2, 3, 4, 5, 6}));
    EXPECT_EQ(result.queries, (std::vector<std::size_t>{7, 8, 9}));
}

TEST(TimeSplitTest, ZeroPercentQueries) {
    auto db = make_db({10, 20, 30, 40});
    auto result = split::split(db, split::SplitType::Time, 0.0);

    EXPECT_EQ(result.database, (std::vector<std::size_t>{0, 1, 2, 3}));
    EXPECT_TRUE(result.queries.empty());
}

TEST(TimeSplitTest, HundredPercentQueries) {
    auto db = make_db({10, 20, 30, 40});
    auto result = split::split(db, split::SplitType::Time, 1.0);

    EXPECT_TRUE(result.database.empty());
    EXPECT_EQ(result.queries, (std::vector<std::size_t>{0, 1, 2, 3}));
}

TEST(TimeSplitTest, EmptyDatabase) {
    auto db = make_db({});
    auto result = split::split(db, split::SplitType::Time, 0.5);

    EXPECT_TRUE(result.database.empty());
    EXPECT_TRUE(result.queries.empty());
}

TEST(TimeSplitTest, InvalidPercentageThrows) {
    auto db = make_db({1, 2, 3});

    EXPECT_THROW(split::split(db, split::SplitType::Time, -0.1), std::invalid_argument);
    EXPECT_THROW(split::split(db, split::SplitType::Time, 1.1), std::invalid_argument);
}