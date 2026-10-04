#pragma once

#include "backend/data/readers/geolife.hpp"

namespace filters {

// Ground distance in metres, altitude is ignored.
// Approximation, accurate enough between nearby points.
double distance_m(const geolife::Point& a, const geolife::Point& b);

}
