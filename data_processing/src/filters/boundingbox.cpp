#include "backend/data/filters/boundingbox.hpp"
#include "backend/data/filters/filters.hpp"

namespace filters {

std::vector<geolife::Segment> split_by_bbox(const geolife::Segment& segment, const BoundingBox& bbox) {
    std::vector<geolife::Segment> bounded_segments;
    geolife::Segment current;
    std::size_t points_outside = 0;

    for (const auto& point : segment) {
        if (!bbox.contains(point)) {
            ++points_outside;
            continue;
        }
        // Short excursions are dropped and the segment continues, longer ones end it.
        if (points_outside > kMaxPointsOutside) {
            create_segment(bounded_segments, current);
        }
        points_outside = 0;
        current.push_back(point);
    }
    create_segment(bounded_segments, current);

    return bounded_segments;
}

}
