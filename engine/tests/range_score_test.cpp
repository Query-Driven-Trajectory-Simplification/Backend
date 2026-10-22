#include <gtest/gtest.h>

#include <cmath>

#include "backend/engine/evaluator/range_score.hpp"

namespace {

using evaluator::range::query_score;

TEST(RangeScore, SameResultsScoreOne) { EXPECT_DOUBLE_EQ(query_score(20, 20), 1.0); }

TEST(RangeScore, NoSimplifiedResultsScoreZero) { EXPECT_DOUBLE_EQ(query_score(20, 0), 0.0); }

TEST(RangeScore, IsTheShareOfOriginalResultsKept) { EXPECT_DOUBLE_EQ(query_score(20, 15), 0.75); }

TEST(RangeScore, MoreThanTheOriginalIsClippedAtZero) {
    EXPECT_DOUBLE_EQ(query_score(10, 30), 0.0);
}

TEST(RangeScore, NanWhenTheOriginalReturnedNothing) {
    EXPECT_TRUE(std::isnan(query_score(0, 0)));
    EXPECT_TRUE(std::isnan(query_score(0, 5)));
}

} // namespace
