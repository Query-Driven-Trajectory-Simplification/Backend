#include "backend/engine/core/simplification.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace core {

TrajectoryDb materialize(const TrajectoryDb &trajectories, const Simplification &simplification) {
    if (simplification.size() != trajectories.size())
        throw std::invalid_argument("materialize: simplification and database sizes differ");

    const std::size_t num_points = simplification.num_points();
    std::vector<std::uint64_t> offsets{0};
    std::vector<std::int32_t> trajectory_ids;
    std::vector<std::int64_t> t;
    std::vector<double> lat;
    std::vector<double> lon;
    offsets.reserve(trajectories.size() + 1);
    trajectory_ids.reserve(trajectories.size());
    t.reserve(num_points);
    lat.reserve(num_points);
    lon.reserve(num_points);

    for (std::size_t i = 0; i < trajectories.size(); ++i) {
        const TrajectoryView trajectory = trajectories.trajectory(i);
        trajectory_ids.push_back(trajectory.trajectory_id);
        for (const std::size_t kept_index : simplification.trajectory(i)) {
            t.push_back(trajectory.t[kept_index]);
            lat.push_back(trajectory.lat[kept_index]);
            lon.push_back(trajectory.lon[kept_index]);
        }
        offsets.push_back(t.size());
    }

    return TrajectoryDb(std::move(offsets), std::move(trajectory_ids), std::move(t), std::move(lat),
                        std::move(lon));
}

} // namespace core
