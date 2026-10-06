#include "backend/data/filters/filters.hpp"
#include "backend/data/filters/boundingbox.hpp"

namespace filters {

std::vector<geolife::Segment> apply_filters(geolife::Segment points) {
    std::vector<geolife::Segment> segments;
    segments.push_back(std::move(points));

    segments = split_by_bbox(segments, kBeijing);

    return segments;
}

}
