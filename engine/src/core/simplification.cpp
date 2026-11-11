#include "backend/engine/core/simplification.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace core {

TrajectoryDb materialize(const TrajectoryDb &trajectories, const Simplification &simplification) {
    if (simplification.size() != trajectories.size())
        throw std::invalid_argument("materialize: simplification and database sizes differ");

    const std::size_t num_points = simplification.num_points();
    TrajectoryDb result;
    result.m_offsets.reserve(trajectories.size() + 1);
    result.m_trajectory_ids.reserve(trajectories.size());
    result.m_t.reserve(num_points);
    result.m_lat.reserve(num_points);
    result.m_lon.reserve(num_points);
    result.m_x.reserve(num_points);
    result.m_y.reserve(num_points);
    result.m_boxes.reserve(trajectories.size());

    constexpr double inf = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < trajectories.size(); ++i) {
        const TrajectoryView trajectory = trajectories.trajectory(i);
        const auto kept = simplification.trajectory(i);
        result.m_trajectory_ids.push_back(trajectory.trajectory_id);
        BoundingBox box{inf, -inf, inf, -inf};
        for (std::size_t j = 0; j < kept.size(); ++j) {
            const auto index = kept[j];
            if (index >= trajectory.size())
                throw std::out_of_range("materialize: kept point index out of range");
            // Preserve strictly increasing timestamps without revalidating the source database.
            if (j > 0 && index <= kept[j - 1])
                throw std::invalid_argument(
                    "materialize: kept indices must be strictly increasing");
            result.m_t.push_back(trajectory.t[index]);
            result.m_lat.push_back(trajectory.lat[index]);
            result.m_lon.push_back(trajectory.lon[index]);
            const double x = trajectory.x[index], y = trajectory.y[index];
            result.m_x.push_back(x);
            result.m_y.push_back(y);
            box.x_min = std::min(box.x_min, x);
            box.x_max = std::max(box.x_max, x);
            box.y_min = std::min(box.y_min, y);
            box.y_max = std::max(box.y_max, y);
        }
        result.m_offsets.push_back(result.m_t.size());
        result.m_boxes.push_back(box);
    }
    return result;
}

} // namespace core
