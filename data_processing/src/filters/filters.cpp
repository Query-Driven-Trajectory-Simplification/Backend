#include "backend/data/filters/filters.hpp"
#include "backend/data/filters/boundingbox.hpp"
#include "backend/data/filters/minsize.hpp"
#include "backend/data/filters/timegap.hpp"

#include <utility>

namespace filters {

std::vector<geolife::Segment> apply_filters(const geolife::Segment& segment) {
    std::vector<geolife::Segment> segments;
    for (const auto& bounded_segment : split_by_bbox(segment, kBeijing)) {
        for (auto& gap_segment : split_by_time_gap(bounded_segment, kMaxGapSeconds)) {
            segments.push_back(std::move(gap_segment));
        }
    }
    return drop_small_segments(std::move(segments), kMinSize);
}

}
