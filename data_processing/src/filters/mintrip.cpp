#include "backend/data/filters/mintrip.hpp"

#include "backend/data/filters/distance.hpp"

namespace filters {

namespace {

double length_m(const geolife::Segment& segment) {
    double length = 0;
    for (std::size_t i = 1; i < segment.size(); ++i) {
        length += distance_m(segment[i - 1], segment[i]);
    }
    return length;
}

}

std::vector<geolife::Segment> drop_short_trips(const std::vector<geolife::Segment>& segments, const MinTripParams& params) {
    std::vector<geolife::Segment> trips;
    for (const auto& segment : segments) {
        if (segment.empty() || segment.size() < params.min_points) continue;
        if (segment.back().t - segment.front().t < params.min_duration_s) continue;
        if (length_m(segment) < params.min_length_m) continue;
        trips.push_back(segment);
    }
    return trips;
}

}
