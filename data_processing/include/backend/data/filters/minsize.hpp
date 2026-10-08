#pragma once

#include <cstddef>
#include <vector>

#include "backend/data/readers/geolife.hpp"

namespace filters {

inline constexpr std::size_t kMinSize = 100;

std::vector<geolife::Segment> drop_small_segments(std::vector<geolife::Segment> segments, std::size_t min_size);

}
