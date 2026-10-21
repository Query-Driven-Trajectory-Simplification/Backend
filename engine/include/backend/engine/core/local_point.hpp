#pragma once

#include <cmath>

// Please don't use this as "trajectory consists of LocalPoint". LocalPoint has no timestamp, so it
// wouldnt make sense to use for the direct points of trajectories, if needed make a Point struct

namespace core {

struct LocalPoint {
    double x;
    double y;
};

inline double distance(const LocalPoint &a, const LocalPoint &b) {
    return std::hypot(a.x - b.x, a.y - b.y);
}

} // namespace core