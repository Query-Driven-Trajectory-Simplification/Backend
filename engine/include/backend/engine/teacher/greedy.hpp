#pragma once

#include "backend/engine/teacher/objective.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace teacher {

// BKH's workload-driven selector for one trajectory: from the endpoints plus a seed, repeatedly
// keep the point that raises the trajectory's range and where-at terms of J_T the most. kNN and
// join are left out, they are too costly here and get scored when the candidates are compared.
// Once no point gains anything, the rest is taken from fill, in order
class Greedy {
  public:
    Greedy(const Objective &objective, std::size_t i);

    // Exactly min(b, n) points, sorted, endpoints and seed included
    std::vector<std::size_t> run(std::size_t b, std::span<const std::size_t> seed,
                                 std::span<const std::size_t> fill) const;

    // The same points in the order they were kept. With the same seed and fill, the first b of
    // sequence(b') are run(b) for any b <= b', so one call serves a whole value curve
    std::vector<std::size_t> sequence(std::size_t limit, std::span<const std::size_t> seed,
                                      std::span<const std::size_t> fill) const;

  private:
    core::TrajectoryView m_v;
    std::vector<std::vector<std::uint32_t>> m_box_points; // points inside each of i's boxes
    std::vector<double> m_box_weight;                     // scaled J_T weight of each box
    std::vector<std::uint64_t> m_offsets;                 // boxes of each point, CSR
    std::vector<std::uint32_t> m_point_boxes;
    std::vector<double> m_probe_t, m_probe_x, m_probe_y; // i's where-at probes, by time
    double m_probe_weight = 0;                           // scaled J_T weight of each probe
    double m_min_tolerance = 0, m_max_tolerance = 0;
};

} // namespace teacher
