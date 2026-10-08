#pragma once

#include <cstddef>
#include <vector>

#include "backend/data/readers/geolife.hpp"

namespace filters {

// Inclusive lat/lon box, altitude is ignored.
struct BoundingBox {
    double min_lat;
    double max_lat;
    double min_lon;
    double max_lon;

    bool contains(const geolife::Point& p) const {
        return p.lat >= min_lat &&
               p.lat <= max_lat &&
               p.lon >= min_lon &&
               p.lon <= max_lon;
    }
};

// Approximate Beijing boundaries, taken from MLsimp boundaries.
inline constexpr BoundingBox kBeijing = {
    .min_lat = 39.4,
    .max_lat = 41.59471,
    .min_lon = 115.7001,
    .max_lon = 117.39994,
};

// Longest run of consecutive outside points that is dropped instead of splitting the segment.
inline constexpr std::size_t kMaxPointsOutside = 5;

// Splits a segment into the runs of points inside bbox, drops short excursions outside bbox.
std::vector<geolife::Segment> split_by_bbox(const geolife::Segment& segment, const BoundingBox& bbox);

}
