#pragma once

#include "backend/engine/teacher/objective.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace teacher {

// BKH's workload-driven selector: from the endpoints and a seed, repeatedly keep the point that
// raises range F1 (other trajectories original) and i's where-at term most, then take the rest
// from fill, skipping kept points
class Greedy {
  public:
    Greedy(const Objective &objective, std::size_t i);

    // Exactly min(budget, n) points, sorted, endpoints and seed included
    std::vector<std::size_t> run(std::size_t budget, std::span<const std::size_t> seed,
                                 std::span<const std::size_t> fill) const;

    // In the order kept: its first b points are run(b)
    std::vector<std::size_t> sequence(std::size_t limit, std::span<const std::size_t> seed,
                                      std::span<const std::size_t> fill) const;

    // Non-nested 1-opt neighbours of a sorted kept set (knot shifts and drop-and-add swaps),
    // preserving budget and endpoints. range_weights: live marginal F1 per box in boxes(i) order
    std::vector<std::vector<std::size_t>> refine_candidates(std::span<const std::size_t> kept,
                                                            std::span<const double> range_weights,
                                                            double range_scale,
                                                            std::span<const double> segments) const;

  private:
    core::TrajectoryView m_trajectory;
    std::vector<std::vector<std::uint32_t>> m_box_points; // points inside each of i's boxes
    std::vector<double> m_box_weight;                     // scaled J_T weight of each box
    std::vector<std::uint64_t> m_offsets;                 // boxes of each point, CSR
    std::vector<std::uint32_t> m_point_boxes;
    std::vector<double> m_probe_t, m_probe_x, m_probe_y; // i's where-at probes, by time
    double m_probe_weight = 0;                           // scaled J_T weight of each probe
    double m_min_tolerance = 0, m_max_tolerance = 0;
};

} // namespace teacher
