#include "backend/data/filters/filters.hpp"
#include "backend/data/filters/boundingbox.hpp"

namespace filters {

std::vector<geolife::Segment> apply_filters(const geolife::Segment& segment) {
    return split_by_bbox(segment, kBeijing);
}

}
