#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/core/projection.hpp"
#include <stdexcept>
#include <string>
#include <utility>
namespace core {

TrajectoryDb::TrajectoryDb(std::vector<std::uint64_t> offsets,
                           std::vector<std::int32_t> trajectory_ids, std::vector<std::int64_t> t,
                           std::vector<double> lat, std::vector<double> lon)
    : m_offsets(std::move(offsets)), m_trajectory_ids(std::move(trajectory_ids)), m_t(std::move(t)),
      m_lat(std::move(lat)), m_lon(std::move(lon)) {
    if (m_lat.size() != m_t.size() || m_lon.size() != m_t.size())
        throw std::invalid_argument("TrajectoryDb: t/lat/lon have different lengths");
    if (m_offsets.size() != m_trajectory_ids.size() + 1 || m_offsets.front() != 0 ||
        m_offsets.back() != m_t.size())
        throw std::invalid_argument("TrajectoryDb: offsets don't match the points");

    for (std::size_t i = 0; i < size(); ++i) {
        for (auto k = m_offsets[i] + 1; k < m_offsets[i + 1]; ++k) {
            if (m_t[k] <= m_t[k - 1])
                throw std::invalid_argument(
                    "TrajectoryDb: timestamps not strictly increasing in trajectory " +
                    std::to_string(m_trajectory_ids[i]));
        }
    }

    m_x.reserve(m_t.size());
    m_y.reserve(m_t.size());
    for (std::size_t k = 0; k < m_t.size(); ++k) {
        const auto p = to_local_metres(m_lat[k], m_lon[k]);
        m_x.push_back(p.x);
        m_y.push_back(p.y);
    }
}

TrajectoryDb subset(const TrajectoryDb &trajectories, std::span<const std::size_t> indices) {
    std::size_t num_points = 0;
    for (const std::size_t i : indices)
        num_points += trajectories.trajectory(i).size();

    std::vector<std::uint64_t> offsets{0};
    std::vector<std::int32_t> trajectory_ids;
    std::vector<std::int64_t> t;
    std::vector<double> lat;
    std::vector<double> lon;
    offsets.reserve(indices.size() + 1);
    trajectory_ids.reserve(indices.size());
    t.reserve(num_points);
    lat.reserve(num_points);
    lon.reserve(num_points);

    for (const std::size_t i : indices) {
        const TrajectoryView trajectory = trajectories.trajectory(i);
        trajectory_ids.push_back(trajectory.trajectory_id);
        t.insert(t.end(), trajectory.t.begin(), trajectory.t.end());
        lat.insert(lat.end(), trajectory.lat.begin(), trajectory.lat.end());
        lon.insert(lon.end(), trajectory.lon.begin(), trajectory.lon.end());
        offsets.push_back(t.size());
    }

    return TrajectoryDb(std::move(offsets), std::move(trajectory_ids), std::move(t), std::move(lat),
                        std::move(lon));
}

} // namespace core