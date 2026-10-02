#include "backend/data/filters/boundingbox.hpp"

namespace filters {

std::vector<geolife::Segment> split_by_bbox(const std::vector<geolife::Segment>& segments, const BoundingBox& bbox) {
    std::vector<geolife::Segment> bounded_segments;
    for (const auto& segment : segments) {
        geolife::Segment current;

        auto flush = [&] {
            if (current.size() >= 2) {
                bounded_segments.push_back(std::move(current));
            }
            current.clear();
        };

        for (const auto& point : segment) {
            if (bbox.contains(point)) {
                current.push_back(point);
            } else {
                flush();
            }
        }
        flush();
    }
    return bounded_segments;
}

}
