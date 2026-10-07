#pragma once

#include <cstdint>
#include <vector>

#include "backend/data/readers/geolife.hpp"

namespace filters {

inline constexpr std::int64_t kMaxGapSeconds = 5 * 60;

std::vector<geolife::Segment> split_by_time_gap(const geolife::Segment& segment, std::int64_t max_gap_s);

}
