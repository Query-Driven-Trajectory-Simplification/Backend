#include "backend/data/filters/boundingbox.hpp"

namespace filters {

std::vector<geolife::Segment> split_by_bbox(const geolife::Segment& segment, const BoundingBox& bbox) {
    std::vector<geolife::Segment> bounded_segments;
        geolife::Segment current;

        auto create_segment = [&] {
            if (!current.empty()) {
                bounded_segments.push_back(std::move(current));
            }
            current.clear();
        };

        for (const auto& point : segment) {
            if (bbox.contains(point)) {
                current.push_back(point);
            } else {
                create_segment();
            }
        }
        create_segment();
    
    return bounded_segments;
}

}
