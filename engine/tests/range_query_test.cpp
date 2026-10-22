#include <gtest/gtest.h>

#include "backend/engine/query/range_query.hpp"

namespace {

query::RangeQuery make_box() {
    return {.trajectory_id = 0,
            .lat = 40.0,
            .lon = 116.0,
            .t = 1000,
            .min_lat = 39.9,
            .max_lat = 40.1,
            .min_lon = 115.9,
            .max_lon = 116.1,
            .t_start = 900,
            .t_end = 1100,
            .results = 0};
}

TEST(InBox, PointInsideIsIn) { EXPECT_TRUE(query::in_box(make_box(), 1000, 40.0, 116.0)); }

TEST(InBox, EdgesAreIncluded) {
    const auto box = make_box();
    EXPECT_TRUE(query::in_box(box, 900, 40.0, 116.0));
    EXPECT_TRUE(query::in_box(box, 1100, 40.0, 116.0));
    EXPECT_TRUE(query::in_box(box, 1000, 39.9, 116.0));
    EXPECT_TRUE(query::in_box(box, 1000, 40.1, 116.0));
    EXPECT_TRUE(query::in_box(box, 1000, 40.0, 115.9));
    EXPECT_TRUE(query::in_box(box, 1000, 40.0, 116.1));
}

TEST(InBox, OutsideInAnyDimensionIsOut) {
    const auto box = make_box();
    EXPECT_FALSE(query::in_box(box, 899, 40.0, 116.0));
    EXPECT_FALSE(query::in_box(box, 1101, 40.0, 116.0));
    EXPECT_FALSE(query::in_box(box, 1000, 39.8, 116.0));
    EXPECT_FALSE(query::in_box(box, 1000, 40.2, 116.0));
    EXPECT_FALSE(query::in_box(box, 1000, 40.0, 115.8));
    EXPECT_FALSE(query::in_box(box, 1000, 40.0, 116.2));
}

} // namespace
