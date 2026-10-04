#pragma once

#include <cstdint>
#include <vector>

#include "backend/data/readers/geolife.hpp"

namespace filters {

// A stay is a run of points that remains within radius_m of its first point
// for at least min_duration_s.
struct StayPointParams {
    double radius_m;
    std::int64_t min_duration_s;
};

inline constexpr StayPointParams kDefaultStay = {
    .radius_m = 100.0,
    .min_duration_s = 5 * 60,
};

// Splits each segment at its stays. The trip before a stay ends at the stay's first point
// and the trip after it starts at the stay's last point, the points in between are dropped.
// Runs with fewer than 2 points are dropped.
std::vector<geolife::Segment> split_by_stay_points(const std::vector<geolife::Segment>& segments, const StayPointParams& params);

}
