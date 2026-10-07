#include "backend/data/filters/filters.hpp"
#include "backend/data/filters/boundingbox.hpp"

namespace filters {

std::vector<geolife::Segment> apply_filters(geolife::Segment points) {
    std::vector<geolife::Segment> segments;
    
    segments = split_by_bbox(std::move(points), kBeijing);

    return segments;
}

}
