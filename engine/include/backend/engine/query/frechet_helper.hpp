#pragma once

#include "backend/engine/core/trajectory_db.hpp"

#include <vector>

namespace query {

// Squared instead of std::hypot, which is slow. Fréchet only takes max and min of distances,
// Square still has order
inline double squared_point_dist(double a_x, double a_y, double b_x, double b_y) {
    const double dx = a_x - b_x;
    const double dy = a_y - b_y;
    return dx * dx + dy * dy;
}

// Squared discrete Fréchet distance. Values below max_squared_dist are exact, anything else is
// at or above it.
double get_squared_frechet_distance(const core::TrajectoryView &query,
                                    const core::TrajectoryView &reference, double max_squared_dist,
                                    std::vector<double> &prev, std::vector<double> &curr);

double squared_endpoint_lower_bound(const core::TrajectoryView &query,
                                    const core::TrajectoryView &reference);

double frechet_distance(const core::TrajectoryView &a, const core::TrajectoryView &b);

// Whether the discrete Fréchet distance is at most max_dist, up to a relative 1e-12 for
// rounding. Stops as soon as it is not.
bool frechet_within(const core::TrajectoryView &a, const core::TrajectoryView &b, double max_dist);

double endpoint_lower_bound(const core::TrajectoryView &a, const core::TrajectoryView &b);

} // namespace query
