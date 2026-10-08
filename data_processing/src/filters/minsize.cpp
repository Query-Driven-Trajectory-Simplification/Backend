#include "backend/data/filters/minsize.hpp"

namespace filters {

std::vector<geolife::Segment> drop_small_segments(std::vector<geolife::Segment> segments, std::size_t min_size) {
    // segments is taken by value, erase in place and return it.
    std::erase_if(segments, [&](const geolife::Segment& segment) {
        return segment.size() < min_size;
    });
    return segments;
}

}
