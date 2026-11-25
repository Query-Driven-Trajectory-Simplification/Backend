#include <gtest/gtest.h>

#include "backend/data/readers/tdrive.hpp"

TEST(ParseTDriveLine, ValidLineHasLongitudeFirst) {
    auto p = tdrive::parse_line("1,2008-02-02 15:36:08,116.51172,39.92123");
    ASSERT_TRUE(p) << p.error();
    EXPECT_DOUBLE_EQ(p->lat, 39.92123);
    EXPECT_DOUBLE_EQ(p->lon, 116.51172);
    EXPECT_EQ(p->t, 1201966568); // 2008-02-02 15:36:08, read as UTC
}

TEST(ParseTDriveLine, InvalidDate) {
    auto p = tdrive::parse_line("1,2008-02-30 15:36:08,116.51172,39.92123");
    ASSERT_FALSE(p);
    EXPECT_EQ(p.error(), "invalid date/time 2008-02-30 15:36:08");
}

TEST(ParseTDriveLine, MissingLatitude) {
    auto p = tdrive::parse_line("1,2008-02-02 15:36:08,116.51172");
    ASSERT_FALSE(p);
    EXPECT_EQ(p.error(), "expected 8 fields, could only read 7");
}
