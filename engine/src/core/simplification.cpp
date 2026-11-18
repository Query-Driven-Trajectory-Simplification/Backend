#include "backend/engine/core/simplification.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace core {

Simplification to_simplification(const KeptSets &kept) {
    Simplification simplification;
    for (const auto &k : kept)
        simplification.add_trajectory(k);
    return simplification;
}

KeptSets to_kept(const Simplification &simplification) {
    KeptSets kept(simplification.size());
    for (std::size_t i = 0; i < simplification.size(); ++i)
        kept[i].assign(simplification.trajectory(i).begin(), simplification.trajectory(i).end());
    return kept;
}

KeptSets endpoints_only(const TrajectoryDb &trajectories) {
    KeptSets kept(trajectories.size());
    for (std::size_t i = 0; i < trajectories.size(); ++i) {
        const std::size_t n = trajectories.trajectory(i).size();
        if (n > 0)
            kept[i].push_back(0);
        if (n > 1)
            kept[i].push_back(n - 1);
    }
    return kept;
}

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

TrajectoryDb materialize(const TrajectoryDb &trajectories, const KeptSets &kept) {
    return materialize(trajectories, to_simplification(kept));
}

} // namespace core
