#pragma once

#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/teacher/workloads.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace teacher {

// Kept point indices per trajectory, sorted, endpoints included
using KeptSets = std::vector<std::vector<std::size_t>>;

struct ObjectiveConfig {
    double min_tolerance_m = 25.0; // where-at average-hit ramp, same as the evaluator
    double max_tolerance_m = 500.0;
};

// A join candidate of one trajectory
struct JoinLink {
    std::uint32_t other;
    std::uint32_t pair; // index into the objective's pair list
    bool positive;
};

// The teacher's objective J_T: an equal mean of four normalised metrics on the teacher's own
// workloads, (metric - floor) / (1 - floor), floor = endpoints only, like J5. Clustering is left
// out. Only seen join pairs are used.
//   kNN       for each query e with i in its top k: Fréchet(original e, simplified i) <= cutoff(e).
//             Competitors stay original
//   range     for each box i is in: some kept point is still inside (recall; kept points are a
//             subset of the original, so there are no false positives)
//   where-at  average-hit ramp over i's probes, averaged per trajectory
//   join      macro F1 over the seen candidate pairs, both sides compressed, like the evaluator:
//             F1_t = 2 hits / (found + expected), averaged over trajectories with any truth or
//             any match. Endpoints-only trajectories match every candidate, so the floor has full
//             recall and the loss is in false positives
// kNN, range and where-at add up over trajectories. Join does not, because a pair's outcome
// depends on both sides; JoinState tracks it exactly for one-at-a-time changes.
class Objective {
  public:
    enum Term : std::size_t { Knn, Range, WhereAt, Join, TermCount };
    using Terms = std::array<double, TermCount>;

    // Keeps references to db and workloads
    Objective(const core::TrajectoryDb &db, const TeacherWorkloads &workloads,
              const ObjectiveConfig &config = {});

    // Scaled kNN + range + where-at of trajectory i, 0 with endpoints only. Sums to their part
    // of J_T
    double separable(std::size_t i, std::span<const std::size_t> kept) const;

    // A one-pass score for choosing i's points on their own: separable plus i's join F1 against
    // a fixed partner state (nullptr = original partners), as if each trajectory carried an
    // equal share of the join metric. An approximation, unlike JoinState
    double independent_value(std::size_t i, std::span<const std::size_t> kept,
                             const KeptSets *partners) const;

    // Exact J_T, partners compressed as in kept. 0 with endpoints only
    double total(const KeptSets &kept) const;

    // Unscaled metric values of a whole state, and their floors
    Terms metrics(const KeptSets &kept) const;
    const Terms &floors() const { return m_floor; }
    double scale(Term m) const { return m_scale[m]; }

    std::span<const JoinLink> links(std::size_t i) const { return m_links[i]; }
    std::size_t pair_count() const { return m_pair_i.size(); }
    std::size_t positive_count() const { return m_positive_count; }
    std::size_t truth(std::size_t t) const { return m_truth[t]; }

    // Whether simplified a and simplified b are within the join threshold
    bool match(std::size_t a, std::span<const std::size_t> kept_a, std::size_t b,
               std::span<const std::size_t> kept_b) const;
    // Same, with b original
    bool match_original(std::size_t a, std::span<const std::size_t> kept_a, std::size_t b) const;

  private:
    struct KnnMembership {
        std::uint32_t query;
        double cutoff;
    };
    struct RangeMembership {
        std::uint32_t box;
        double weight; // 1 / (|reference| * scored boxes)
    };

    Terms raw_separable(std::size_t i, std::span<const std::size_t> kept) const;
    double join_metric(const KeptSets &kept) const;

    const core::TrajectoryDb &m_db;
    const TeacherWorkloads &m_workloads;
    ObjectiveConfig m_config;

    std::vector<std::vector<KnnMembership>> m_knn;
    double m_knn_weight = 0; // 1 / (k * valid queries)
    std::vector<std::vector<RangeMembership>> m_range;
    std::vector<std::vector<std::uint32_t>> m_probes; // where-at query indices per trajectory
    double m_where_at_weight = 0;                     // 1 / trajectories with probes

    std::vector<std::uint32_t> m_pair_i, m_pair_j; // seen pairs, positives first
    std::size_t m_positive_count = 0;
    std::vector<std::vector<JoinLink>> m_links; // per trajectory
    std::vector<std::size_t> m_truth;           // seen positives per trajectory
    std::size_t m_with_truth = 0;               // trajectories with any seen positive

    Terms m_floor{};          // global floor of each metric
    Terms m_scale{};          // 1 / (TermCount * (1 - floor))
    std::vector<double> m_v0; // scaled separable terms of each trajectory with endpoints only
};

// The exact join metric under one-at-a-time changes. Keeps every seen pair's outcome and every
// trajectory's hits and found counts, so changing i only rescores i's pairs
class JoinState {
  public:
    JoinState(const Objective &objective, const KeptSets &state);

    // Change of the scaled join term of J_T if i switched to kept
    double delta(std::size_t i, std::span<const std::size_t> kept, const KeptSets &state) const;
    // Records i's switch; state must already hold i's new kept set
    void apply(std::size_t i, const KeptSets &state);

    double metric() const; // macro F1 over the seen pairs

  private:
    bool counted(std::size_t t, std::size_t found) const;
    double f1(std::size_t t, std::size_t hits, std::size_t found) const;

    const Objective &m_objective;
    std::vector<char> m_matched; // per pair
    std::vector<std::size_t> m_hits, m_found;
    double m_sum = 0; // sum of F1 over counted trajectories
    std::size_t m_counted = 0;
};

} // namespace teacher
