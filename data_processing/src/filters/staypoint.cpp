#include "backend/data/filters/staypoint.hpp"

#include <cstddef>

#include "backend/data/filters/distance.hpp"

namespace filters {

std::vector<geolife::Segment> split_by_stay_points(const std::vector<geolife::Segment>& segments, const StayPointParams& params) {
    std::vector<geolife::Segment> trips;
    for (const auto& segment : segments) {
        geolife::Segment current;

        // If trip has at least two points, keep it.
        auto flush = [&] {
            if (current.size() >= 2) {
                trips.push_back(std::move(current));
            }
            current.clear();
        };

        // Starting from point i, move point j forward until outsite radius.
        std::size_t i = 0;
        while (i < segment.size()) {
            // [i, j) is the run of points that stays within the radius of point i.
            std::size_t j = i + 1;
            while (j < segment.size() && distance_m(segment[i], segment[j]) <= params.radius_m) {
                ++j;
            }

            // Trip must last at least min_duration_s.
            if (segment[j - 1].t - segment[i].t >= params.min_duration_s) {
                current.push_back(segment[i]);
                flush();
                current.push_back(segment[j - 1]);
                i = j;
            } else {
                current.push_back(segment[i]);
                ++i;
            }
        }
        flush();
    }
    return trips;
}

}
