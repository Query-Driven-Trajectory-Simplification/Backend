#pragma once

#include "backend/engine/teacher/objective.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace teacher {

// BKH's workload-driven selector: from the endpoints, repeatedly keep the point that raises
// range F1 (other trajectories original) and i's where-at term most, then take the rest from
// fill, skipping kept points
class Greedy {
  public:
    Greedy(const Objective &objective, std::size_t i);

    // Exactly min(budget, n) points, sorted, endpoints included
    std::vector<std::size_t> run(std::size_t budget, std::span<const std::size_t> fill) const;

    // In the order kept: its first b points are run(b)
    std::vector<std::size_t> sequence(std::size_t limit, std::span<const std::size_t> fill) const;

    // Adaptive joint greedy sequence: at each step picks the point maximizing exact
    // Range + Where-At + Shape (max + arc) + Margin (kNN + Join) across Range+WA and segment splits
    std::vector<std::size_t> joint_sequence(std::size_t limit, const Objective &objective,
                                            std::size_t i, std::span<const double> segments,
                                            std::span<const std::size_t> fill) const;

    // Non-nested knot-shift neighbours of a sorted kept set (Range+WA, simultaneous red-black, and
    // Fréchet peak/midpoint shifts), preserving budget and endpoints. range_weights: live marginal
    // F1 per box in boxes(i) order
    std::vector<std::vector<std::size_t>> refine_candidates(std::span<const std::size_t> kept,
                                                            std::span<const double> range_weights,
                                                            double range_scale,
                                                            std::span<const double> segments) const;

    struct PointTradeOffers {
        std::vector<std::vector<std::size_t>> drops; // (b - 1)-point sets
        std::vector<std::vector<std::size_t>> adds;  // (b + 1)-point sets
    };
    // Candidate 1-point drops and 1-point additions directly on a refined sorted kept set
    PointTradeOffers point_trade_candidates(std::span<const std::size_t> kept,
                                            std::span<const double> range_weights,
                                            double range_scale,
                                            std::span<const double> segments) const;

  private:
    struct SegGeom {
        double max_sq = 0.0;
        std::size_t worst = 0;
        double rms_num = 0.0;
        double arc_shape_num = 0.0;
    };
    struct LocalContext;
    double score_probe(std::size_t p, std::size_t from, std::size_t to, double near_sq,
                       double far_sq) const;
    std::pair<std::size_t, std::size_t> probes_between(std::size_t from, std::size_t to) const;
    std::vector<double> build_arc(std::span<const double> segments) const;
    SegGeom eval_seg_geom(std::span<const double> arc, std::size_t from, std::size_t to,
                          double near_sq, double far_sq) const;

    core::TrajectoryView m_trajectory;
    std::vector<std::vector<std::uint32_t>> m_box_points; // points inside each of i's boxes
    std::vector<double> m_box_weight;                     // scaled J_T weight of each box
    std::vector<std::uint64_t> m_offsets;                 // boxes of each point, CSR
    std::vector<std::uint32_t> m_point_boxes;
    std::vector<double> m_probe_t, m_probe_x, m_probe_y; // i's where-at probes, by time
    double m_probe_weight = 0;                           // scaled J_T weight of each probe
    double m_min_tolerance = 0, m_max_tolerance = 0;
    double m_min_shape = 25.0, m_max_shape = 200.0;
    double m_arc_shape_weight = 0; // scaled J_T weight of the arc-shape half of Shape
};

} // namespace teacher
