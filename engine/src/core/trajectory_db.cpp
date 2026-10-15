#include "backend/engine/core/trajectory_db.hpp"
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
                throw std::invalid_argument("TrajectoryDb: timestamps not strictly increasing in trajectory "
                                            + std::to_string(m_trajectory_ids[i]));
        }
    }
}

} // namespace core