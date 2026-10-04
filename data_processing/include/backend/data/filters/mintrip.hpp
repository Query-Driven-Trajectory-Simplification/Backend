#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "backend/data/readers/geolife.hpp"

namespace filters {

// A trip must reach all three minimums to be kept.
struct MinTripParams {
    std::size_t min_points;
    std::int64_t min_duration_s;  // last point's time minus first point's time
    double min_length_m;          // summed distance between consecutive points
};

// Set to 0 to disable requirement.
inline constexpr MinTripParams kDefaultMinTrip = {
    .min_points = 10,
    .min_duration_s = 60,
    .min_length_m = 200.0,
};

// Drops every segment that is below any of the minimums. Kept segments are unchanged.
std::vector<geolife::Segment> drop_short_trips(const std::vector<geolife::Segment>& segments, const MinTripParams& params);

}
