#pragma once

#include "backend/engine/core/trajectory_db.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

// example
//core::Simplification s;
// m_offsets = [0]              m_kept = []

//s.add_trajectory({0, 4, 9});
// m_offsets = [0, 3]           m_kept = [0, 4, 9]

//s.add_trajectory({0, 4});
// m_offsets = [0, 3, 5]        m_kept = [0, 4, 9, 0, 4]

//s.add_trajectory({0, 2, 5, 7});
// m_offsets = [0, 3, 5, 9]     m_kept = [0, 4, 9, 0, 4, 0, 2, 5, 7]


namespace core {

// The points kept from each trajectory of a database, as sorted indices into that trajectory
class Simplification {
  public:
    void add_trajectory(std::span<const std::size_t> kept_indices) {
        m_kept.insert(m_kept.end(), kept_indices.begin(), kept_indices.end());
        m_offsets.push_back(m_kept.size());
    }

    std::size_t size() const { return m_offsets.size() - 1; }
    std::size_t num_points() const { return m_kept.size(); }

    std::span<const std::size_t> trajectory(std::size_t i) const {
        assert(i < size());
        return {m_kept.data() + m_offsets[i], m_offsets[i + 1] - m_offsets[i]};
    }

  private:
    std::vector<std::uint64_t> m_offsets{0};
    std::vector<std::size_t> m_kept;
};

// Builds a database holding only the kept points, with the same trajectories in the same order
TrajectoryDb materialize(const TrajectoryDb &trajectories, const Simplification &simplification);

} // namespace core