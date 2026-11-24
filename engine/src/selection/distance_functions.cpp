#include "backend/engine/selection/distance_functions.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <span>

namespace selection::distance {

namespace {

// Heading of a -> b in [0, 2 pi); atan2(0, 0) = 0 for repeated points
double heading(double dx, double dy) {
    const double angle = std::atan2(dy, dx);
    return angle >= 0.0 ? angle : 2.0 * std::numbers::pi + angle;
}

} // namespace

double max_sed_squared(const core::TrajectoryView &trajectory, std::size_t first,
                       std::size_t last) {
    const std::span<const std::int64_t> ts = trajectory.t;
    const std::span<const double> xs = trajectory.x;
    const std::span<const double> ys = trajectory.y;

    const double t0 = static_cast<double>(ts[first]);
    const double inv_dt = 1.0 / (static_cast<double>(ts[last]) - t0);
    const double x0 = xs[first];
    const double y0 = ys[first];
    const double seg_dx = xs[last] - x0;
    const double seg_dy = ys[last] - y0;

    double error = 0.0;
    for (std::size_t k = first + 1; k < last; ++k) {
        const double c = (static_cast<double>(ts[k]) - t0) * inv_dt;
        const double dx = x0 + c * seg_dx - xs[k];
        const double dy = y0 + c * seg_dy - ys[k];
        error = std::max(error, dx * dx + dy * dy);
    }
    return error;
}

double max_ped_squared(const core::TrajectoryView &trajectory, std::size_t first,
                       std::size_t last) {
    const std::span<const double> xs = trajectory.x;
    const std::span<const double> ys = trajectory.y;
    const double x0 = xs[first];
    const double y0 = ys[first];
    const double seg_dx = xs[last] - x0;
    const double seg_dy = ys[last] - y0;
    const double length_sq = seg_dx * seg_dx + seg_dy * seg_dy;
    if (length_sq == 0.0)
        return 0.0;
    double cross_sq = 0.0;
    for (std::size_t k = first + 1; k < last; ++k) {
        const double cross = seg_dx * (ys[k] - y0) - seg_dy * (xs[k] - x0);
        cross_sq = std::max(cross_sq, cross * cross);
    }
    return cross_sq / length_sq;
}

std::vector<double> segment_headings(const core::TrajectoryView &trajectory) {
    const std::size_t n = trajectory.size();
    std::vector<double> headings(n > 0 ? n - 1 : 0);
    for (std::size_t k = 0; k + 1 < n; ++k)
        headings[k] =
            heading(trajectory.x[k + 1] - trajectory.x[k], trajectory.y[k + 1] - trajectory.y[k]);
    return headings;
}

double max_heading_change(const core::TrajectoryView &trajectory,
                          const std::vector<double> &headings, std::size_t first,
                          std::size_t last) {
    const double theta =
        heading(trajectory.x[last] - trajectory.x[first], trajectory.y[last] - trajectory.y[first]);
    double error = 0.0;
    for (std::size_t k = first; k < last; ++k) {
        const double gap = std::abs(theta - headings[k]);
        error = std::max(error, std::min(gap, 2.0 * std::numbers::pi - gap));
    }
    return error;
}

} // namespace selection::distance
