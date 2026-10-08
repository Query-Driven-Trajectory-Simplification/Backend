#include "backend/data/filters/timegap.hpp"
#include "backend/data/filters/filters.hpp"

namespace filters {

std::vector<geolife::Segment> split_by_time_gap(const geolife::Segment& segment, std::int64_t max_gap_s) {
    std::vector<geolife::Segment> gap_segments;
    geolife::Segment current;

    for (const auto& point : segment) {
        if (!current.empty() && point.t - current.back().t > max_gap_s) {
            create_segment(gap_segments, current);
        }
        current.push_back(point);
    }
    create_segment(gap_segments, current);

    return gap_segments;
}

}
