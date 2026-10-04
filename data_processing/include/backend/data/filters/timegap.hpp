#pragma once

#include <cstdint>
#include <vector>

#include "backend/data/readers/geolife.hpp"

namespace filters {

// Longest pause between two consecutive points of the same trip.
inline constexpr std::int64_t kMaxGapSeconds = 5 * 60;

// Splits each segment wherever two consecutive points are more than max_gap_s apart.
// Runs with fewer than 2 points are dropped.
std::vector<geolife::Segment> split_by_time_gap(const std::vector<geolife::Segment>& segments, std::int64_t max_gap_s);

}
