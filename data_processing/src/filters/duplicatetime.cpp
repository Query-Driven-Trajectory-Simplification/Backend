#include "backend/data/filters/duplicatetime.hpp"

#include <algorithm>

namespace filters {

geolife::Segment drop_duplicate_times(geolife::Segment segment) {
    const auto repeated = std::ranges::unique(segment, {}, &geolife::Point::t);
    segment.erase(repeated.begin(), repeated.end());
    return segment;
}

}
