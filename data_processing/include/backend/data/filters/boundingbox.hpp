#pragma once

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

// Approximate extent of Beijing municipality.
inline constexpr BoundingBox kBeijing = {
    .min_lat = 39.44,
    .max_lat = 41.06,
    .min_lon = 115.42,
    .max_lon = 117.50,
};

// Splits each segment into the runs of consecutive points inside bbox.
// Points outside are dropped, as are runs with fewer than 2 points.
std::vector<geolife::Segment> split_by_bbox(const std::vector<geolife::Segment>& segments, const BoundingBox& bbox);

}
