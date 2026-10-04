#include "backend/data/filters/distance.hpp"

#include <cmath>
#include <numbers>

namespace filters {

namespace {

constexpr double kEarthRadiusM = 6'371'000.0;
constexpr double kDegToRad = std::numbers::pi / 180.0;

}

// Approximation: treats the area between a and b as flat.
double distance_m(const geolife::Point& a, const geolife::Point& b) {
    const double mean_lat = (a.lat + b.lat) / 2 * kDegToRad;
    // A degree of longitude shrinks by cos(latitude) away from the equator.
    const double x = (b.lon - a.lon) * kDegToRad * std::cos(mean_lat);
    const double y = (b.lat - a.lat) * kDegToRad;
    // Pythagoras on the angles (radians), scaled to metres by the Earth's radius.
    return kEarthRadiusM * std::sqrt(x * x + y * y);
}

}
