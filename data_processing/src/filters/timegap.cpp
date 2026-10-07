#include "backend/data/filters/timegap.hpp"

namespace filters {

std::vector<geolife::Segment> split_by_time_gap(const geolife::Segment& segment, std::int64_t max_gap_s) {
    std::vector<geolife::Segment> gap_segments;
    geolife::Segment current;

    auto create_segment = [&] {
        if (!current.empty()) {
            gap_segments.push_back(std::move(current));
        }
        current.clear();
    };

    for (const auto& point : segment) {
        if (!current.empty() && point.t - current.back().t > max_gap_s) {
            create_segment();
        }
        current.push_back(point);
    }
    create_segment();

    return gap_segments;
}

}
