#include "backend/engine/selection/sed.hpp"

#include "backend/engine/selection/bottom_up.hpp"

#include <cmath>
#include <cstdint>
#include <span>

namespace selection::sed {

double synchronous_distance(const core::TrajectoryView &trajectory, std::size_t first,
                            std::size_t last, std::size_t point) {
    const std::span<const std::int64_t> ts = trajectory.t;
    const std::span<const double> xs = trajectory.x;
    const std::span<const double> ys = trajectory.y;

    const double c = (static_cast<double>(ts[point]) - static_cast<double>(ts[first])) /
                     (static_cast<double>(ts[last]) - static_cast<double>(ts[first]));

    const double dx = xs[first] + c * (xs[last] - xs[first]) - xs[point];
    const double dy = ys[first] + c * (ys[last] - ys[first]) - ys[point];
    return std::sqrt(dx * dx + dy * dy);
}

std::vector<std::size_t> simplify(const core::TrajectoryView &trajectory,
                                  std::size_t points_to_keep) {
    return bottom_up::simplify(trajectory, points_to_keep, bottom_up::Measure::Sed);
}

std::vector<std::size_t> sequence(const core::TrajectoryView &trajectory, std::size_t limit) {
    return bottom_up::sequence(trajectory, limit, bottom_up::Measure::Sed);
}

} // namespace selection::sed
