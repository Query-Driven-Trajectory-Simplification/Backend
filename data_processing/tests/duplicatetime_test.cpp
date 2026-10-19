#include <gtest/gtest.h>

#include "backend/data/filters/duplicatetime.hpp"

TEST(DropDuplicateTimes, KeepsFirstPointOfEachSecond) {
    const geolife::Segment segment = {{39.0, 116.0, 10},
                                      {39.1, 116.1, 10},
                                      {39.2, 116.2, 11},
                                      {39.3, 116.3, 11},
                                      {39.4, 116.4, 12}};
    const auto result = filters::drop_duplicate_times(segment);
    ASSERT_EQ(result.size(), 3u);
    EXPECT_EQ(result[0].t, 10);
    EXPECT_DOUBLE_EQ(result[0].lat, 39.0);
    EXPECT_EQ(result[1].t, 11);
    EXPECT_DOUBLE_EQ(result[1].lat, 39.2);
    EXPECT_EQ(result[2].t, 12);
}

TEST(DropDuplicateTimes, IncreasingTimesUnchanged) {
    const geolife::Segment segment = {{39.0, 116.0, 1}, {39.1, 116.1, 2}, {39.2, 116.2, 3}};
    EXPECT_EQ(filters::drop_duplicate_times(segment).size(), 3u);
}

TEST(DropDuplicateTimes, EmptySegment) { EXPECT_TRUE(filters::drop_duplicate_times({}).empty()); }
