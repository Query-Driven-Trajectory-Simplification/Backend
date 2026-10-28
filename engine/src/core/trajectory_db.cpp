#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/core/projection.hpp"
#include <algorithm>
#include <limits>
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

    const auto count = m_t.size();
    m_x.resize(count);
    m_y.resize(count);
    const auto *latitudes = m_lat.data();
    const auto *longitudes = m_lon.data();
    auto *xs = m_x.data();
    auto *ys = m_y.data();
    for (std::size_t k = 0; k < count; ++k) {
        const auto p = to_local_metres(latitudes[k], longitudes[k]);
        xs[k] = p.x;
        ys[k] = p.y;
    }

    // For each trajectory it goes throguh all its points
    // and keeps min/max of x and y coordinates seen
    constexpr double inf = std::numeric_limits<double>::infinity();
    m_boxes.resize(size());
    for (std::size_t i = 0; i < size(); ++i) {
        BoundingBox box{inf, -inf, inf, -inf};
        for (auto k = m_offsets[i]; k < m_offsets[i + 1]; ++k) {
            box.x_min = std::min(box.x_min, xs[k]);
            box.x_max = std::max(box.x_max, xs[k]);
            box.y_min = std::min(box.y_min, ys[k]);
            box.y_max = std::max(box.y_max, ys[k]);
        }
        m_boxes[i] = box;
    }
}

TrajectoryDb subset(const TrajectoryDb &trajectories, std::span<const std::size_t> indices) {
    std::size_t num_points = 0;
    for (const std::size_t i : indices) {
        if (i >= trajectories.size())
            throw std::out_of_range("subset: trajectory index out of range");
        num_points += trajectories.trajectory(i).size();
    }

    TrajectoryDb result;
    result.m_offsets.reserve(indices.size() + 1);
    result.m_trajectory_ids.reserve(indices.size());
    result.m_t.reserve(num_points);
    result.m_lat.reserve(num_points);
    result.m_lon.reserve(num_points);
    result.m_x.reserve(num_points);
    result.m_y.reserve(num_points);
    result.m_boxes.reserve(indices.size());

    for (const std::size_t i : indices) {
        const auto trajectory = trajectories.trajectory(i);
        result.m_trajectory_ids.push_back(trajectory.trajectory_id);
        result.m_boxes.push_back(trajectory.bbox);
        result.m_t.insert(result.m_t.end(), trajectory.t.begin(), trajectory.t.end());
        result.m_lat.insert(result.m_lat.end(), trajectory.lat.begin(), trajectory.lat.end());
        result.m_lon.insert(result.m_lon.end(), trajectory.lon.begin(), trajectory.lon.end());
        result.m_x.insert(result.m_x.end(), trajectory.x.begin(), trajectory.x.end());
        result.m_y.insert(result.m_y.end(), trajectory.y.begin(), trajectory.y.end());
        result.m_offsets.push_back(result.m_t.size());
    }
    return result;
}

} // namespace core