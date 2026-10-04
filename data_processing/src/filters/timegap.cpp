#include "backend/data/filters/timegap.hpp"

namespace filters {

std::vector<geolife::Segment> split_by_time_gap(const std::vector<geolife::Segment>& segments, std::int64_t max_gap_s) {
    std::vector<geolife::Segment> trips;
    for (const auto& segment : segments) {
        geolife::Segment current;

        auto flush = [&] {
            if (current.size() >= 2) {
                trips.push_back(std::move(current));
            }
            current.clear();
        };

        for (const auto& point : segment) {
            if (!current.empty() && point.t - current.back().t > max_gap_s) {
                flush();
            }
            current.push_back(point);
        }
        flush();
    }
    return trips;
}

}
