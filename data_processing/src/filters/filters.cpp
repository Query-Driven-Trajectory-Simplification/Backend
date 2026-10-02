#include "backend/data/filters/filters.hpp"

namespace filters {

std::vector<geolife::Segment> apply_filters(const geolife::Segment& points) {
    std::vector<geolife::Segment> segments = {points};

    return segments;
}

}
