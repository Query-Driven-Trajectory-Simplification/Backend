#pragma once

#include <vector>

#include "backend/data/readers/geolife.hpp"

namespace filters {

// Runs all filters on one trajectory and returns the segments that remain.
std::vector<geolife::Segment> apply_filters(geolife::Segment points);

}
