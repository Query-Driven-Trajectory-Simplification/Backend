#include "backend/engine/core/interpolation.hpp"

#include <algorithm>
#include <cstddef>

namespace core {

LocalPoint position_at(const TrajectoryView &trajectory, double t) {
    const auto timestamps = trajectory.t;

    const std::size_t last = timestamps.size() - 1;
    if (t >= static_cast<double>(timestamps[last]))
        return {trajectory.x[last], trajectory.y[last]};

    const auto first_after =
        std::upper_bound(timestamps.begin(), timestamps.end(), t); // uses binary search
    const auto after = static_cast<std::size_t>(first_after - timestamps.begin());
    const std::size_t before = after - 1;

    // How far t is from before to after: 0 = at before, 1 = at after
    const double t_before = static_cast<double>(timestamps[before]);
    const double t_after = static_cast<double>(timestamps[after]);
    const double fraction = (t - t_before) / (t_after - t_before);

    return {
        .x = trajectory.x[before] + fraction * (trajectory.x[after] - trajectory.x[before]),
        .y = trajectory.y[before] + fraction * (trajectory.y[after] - trajectory.y[before]),
    };
}

} // namespace core