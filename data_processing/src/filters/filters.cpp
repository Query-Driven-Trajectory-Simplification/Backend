#include "backend/data/filters/filters.hpp"
#include "backend/data/filters/boundingbox.hpp"

namespace filters {

std::vector<geolife::Segment> apply_filters(const geolife::Segment& points) {
    std::vector<geolife::Segment> segments = {points};
    segments = split_by_bbox(segments, kBeijing);

    return segments;
}

}
