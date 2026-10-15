#pragma once

#include <cassert>
#include <cstdint>
#include <span>
#include <vector>

namespace core {

struct TrajectoryView {
    std::int32_t trajectory_id;
    std::span<const std::int64_t> t;
    std::span<const double> lat;
    std::span<const double> lon;

    std::size_t size() const { return t.size(); }
};

class TrajectoryDb {
  public:
    TrajectoryDb(std::vector<std::uint64_t> offsets, std::vector<std::int32_t> trajectory_ids,
                 std::vector<std::int64_t> t, std::vector<double> lat, std::vector<double> lon);

    std::size_t size() const { return m_trajectory_ids.size(); }
    std::size_t num_points() const { return m_t.size(); }

    TrajectoryView trajectory(std::size_t i) const {
        assert(i < size());
        const auto start = m_offsets[i];
        const auto len = m_offsets[i + 1] - start;
        return TrajectoryView{m_trajectory_ids[i],
                              {m_t.data() + start, len},
                              {m_lat.data() + start, len},
                              {m_lon.data() + start, len}};
    }

  private:
    std::vector<std::uint64_t> m_offsets;
    std::vector<std::int32_t> m_trajectory_ids;
    std::vector<std::int64_t> m_t;
    std::vector<double> m_lat;
    std::vector<double> m_lon;
};

} // namespace core