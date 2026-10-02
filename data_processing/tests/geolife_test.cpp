#include <cmath>

#include <gtest/gtest.h>

#include "backend/data/readers/geolife.hpp"

TEST(ParsePltLine, ValidLine) {
    auto p = geolife::parse_plt_line("39.984702,116.318417,0,492,39744.1201851852,2008-10-23,02:53:04");
    ASSERT_TRUE(p) << p.error();
    EXPECT_DOUBLE_EQ(p->lat, 39.984702);
    EXPECT_DOUBLE_EQ(p->lon, 116.318417);
    EXPECT_DOUBLE_EQ(p->alt_m, 492 * 0.3048);
    EXPECT_EQ(p->t, 1224730384);
}

TEST(ParsePltLine, UnknownAltitudeIsNaN) {
    auto p = geolife::parse_plt_line("39.984702,116.318417,0,-777,39744.1201851852,2008-10-23,02:53:04");
    ASSERT_TRUE(p) << p.error();
    EXPECT_TRUE(std::isnan(p->alt_m));
}

TEST(ParsePltLine, InvalidDate) {
    auto p = geolife::parse_plt_line("39.984702,116.318417,0,492,39744.1201851852,2008-02-30,02:53:04");
    ASSERT_FALSE(p);
    EXPECT_EQ(p.error(), "invalid date/time 2008-02-30 02:53:04"); // 30th feb doesnt exist
}

TEST(ParsePltLine, InvalidTime) {
    auto p = geolife::parse_plt_line("39.984702,116.318417,0,492,39744.1201851852,2008-10-23,25:00:00");
    ASSERT_FALSE(p);
    EXPECT_EQ(p.error(), "invalid date/time 2008-10-23 25:00:00");
}

TEST(ParsePltLine, BadLongitude) {
    auto p = geolife::parse_plt_line("39.984702,abc,0,492,39744.1201851852,2008-10-23,02:53:04");
    ASSERT_FALSE(p);
    EXPECT_EQ(p.error(), "expected 9 fields, could only read 1");
}
