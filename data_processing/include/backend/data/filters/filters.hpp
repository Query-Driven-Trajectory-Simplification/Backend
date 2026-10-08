#pragma once

#include <vector>

#include "backend/data/readers/geolife.hpp"

namespace filters {

// Moves current into segments if it has any points, leaving current empty.
void create_segment(std::vector<geolife::Segment>& segments, geolife::Segment& current);

// Runs all filters on one trajectory and returns the segments that remain.
std::vector<geolife::Segment> apply_filters(const geolife::Segment& segment);

}
