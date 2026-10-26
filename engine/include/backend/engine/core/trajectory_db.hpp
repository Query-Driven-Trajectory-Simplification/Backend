#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace core {

// Smallest axis-aligned box around a trajectory's x/y.
struct BoundingBox {
    double x_min = -std::numeric_limits<double>::infinity();
    double x_max = std::numeric_limits<double>::infinity();
    double y_min = -std::numeric_limits<double>::infinity();
    double y_max = std::numeric_limits<double>::infinity();
};

struct TrajectoryView {
    std::int32_t trajectory_id;
    std::span<const std::int64_t> t;
    std::span<const double> lat;
    std::span<const double> lon;
    std::span<const double> x;
    std::span<const double> y;
    BoundingBox bbox{};

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
        return TrajectoryView{
            m_trajectory_ids[i],
            {m_t.data() + start, len},
            {m_lat.data() + start, len},
            {m_lon.data() + start, len},
            {m_x.data() + start, len},
            {m_y.data() + start, len},
            m_boxes[i],
        };
    }

  private:
    TrajectoryDb() = default;
    friend TrajectoryDb subset(const TrajectoryDb &, std::span<const std::size_t>);

    std::vector<std::uint64_t> m_offsets{0};
    std::vector<std::int32_t> m_trajectory_ids;
    std::vector<std::int64_t> m_t;
    std::vector<double> m_lat;
    std::vector<double> m_lon;
    std::vector<double> m_x;
    std::vector<double> m_y;
    std::vector<BoundingBox> m_boxes;
};

// Copies validated trajectories, including projected coordinates, in the given order.
// Throws std::out_of_range for invalid indices.
TrajectoryDb subset(const TrajectoryDb &trajectories, std::span<const std::size_t> indices);

} // namespace core