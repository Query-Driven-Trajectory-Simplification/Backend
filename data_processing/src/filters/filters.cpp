#include "backend/data/filters/filters.hpp"
#include "backend/data/filters/boundingbox.hpp"
#include "backend/data/filters/minsize.hpp"
#include "backend/data/filters/timegap.hpp"

#include <utility>

namespace filters {

namespace {
// Runs a per-segment split filter on every segment and flattens the results.
template <typename filter_type, typename... Args>
std::vector<geolife::Segment> split_each(const std::vector<geolife::Segment> &segments,
                                         filter_type filter, const Args &...args) {
    std::vector<geolife::Segment> result;
    for (const auto &segment : segments) {
        for (auto &part : filter(segment, args...)) {
            result.push_back(std::move(part));
        }
    }
    return result;
}

} // namespace

std::vector<geolife::Segment> apply_filters(const geolife::Segment &segment) {
    std::vector<geolife::Segment> segments = {segment};
    segments = split_each(segments, split_by_bbox, kBeijing);
    segments = split_each(segments, split_by_time_gap, kMaxGapSeconds);
    segments = drop_small_segments(std::move(segments), kMinSize);
    return segments;
}

} // namespace filters
