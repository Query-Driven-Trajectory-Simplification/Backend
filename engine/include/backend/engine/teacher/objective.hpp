#pragma once

#include "backend/engine/core/trajectory_db.hpp"
#include "backend/engine/teacher/workloads.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace teacher {

// Kept point indices per trajectory, sorted, endpoints included
using KeptSets = std::vector<std::vector<std::size_t>>;

// Owns only the coordinates and timestamps needed by scoring. Views are created on demand,
// so moving or copying the owner never leaves cached spans pointing at another object.
class PreparedTrajectory {
  public:
    PreparedTrajectory(const core::TrajectoryView &original, std::span<const std::size_t> kept);
    core::TrajectoryView view() const { return {m_id, m_t, {}, {}, m_x, m_y, m_box}; }

  private:
    std::int32_t m_id;
    std::vector<std::int64_t> m_t;
    std::vector<double> m_x, m_y;
    core::BoundingBox m_box;
};

class PreparedTrajectories {
  public:
    PreparedTrajectories(const core::TrajectoryDb &db, const KeptSets &kept);
    // The returned views borrow from this owner, which must remain alive and unmodified.
    std::vector<core::TrajectoryView> views() const;

  private:
    std::vector<PreparedTrajectory> m_trajectories;
};

struct ObjectiveConfig {
    double min_tolerance_m = 25.0; // where-at average-hit ramp, same as the evaluator
    double max_tolerance_m = 500.0;
    // kNN hit when i stays within cutoff(e) - Fréchet(e, original i) of its own original, so it
    // can't be fitted to the teacher's queries. Off: too strict, it lowered test J5 by .03
    bool knn_by_shape = false;
    // Every 150 m pair, as the compressor knows the whole database. False: only the seen half, to
    // measure how much of the join gain is fitting those specific pairs
    bool all_join_pairs = true;
    // Weight of the shape term, next to the four metrics' 1/4 each. 0 = off. It rewards keeping
    // the original shape whatever the query, for kNN queries and join thresholds not trained on
    double shape_weight = 0.0;
};

// A join candidate of one trajectory
struct JoinLink {
    std::uint32_t other;
    std::uint32_t pair; // index into the objective's pair list
    bool positive;
};

// The teacher's objective J_T: an equal mean of four normalised metrics on the teacher's own
// workloads, (metric - floor) / (1 - floor), floor = endpoints only, like J5. Clustering is left
// out. Join uses all pairs, or only the seen half (ObjectiveConfig::all_join_pairs).
//   kNN       for each query e with i in its top k: Fréchet(original e, simplified i) <= cutoff(e).
//             Competitors stay original
//   range     for each box i is in: some kept point is still inside (recall; kept points are a
//             subset of the original, so there are no false positives)
//   where-at  average-hit ramp over i's probes, averaged per trajectory
//   join      macro F1 over the candidate pairs, both sides compressed, like the evaluator:
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
    double separable(std::size_t i, const core::TrajectoryView &candidate) const;

    // How closely i's original points stay to the simplified line: the mean over the points of
    // the where-at ramp of each point's distance to the simplified segment spanning its time.
    // The spatial counterpart of where-at, in [0, 1]
    double shape(std::size_t i, const core::TrajectoryView &simplified) const;

    // A one-pass score for choosing i's points on their own: separable plus i's join F1 against
    // a fixed partner state (empty = original partners), as if each trajectory carried an
    // equal share of the join metric. An approximation, unlike JoinState.
    // Exact whenever the value is above beat. Otherwise it may return any value at most beat,
    // skipping the kNN checks that cannot lift it over, the costly part
    double independent_value(std::size_t i, const core::TrajectoryView &candidate,
                             std::span<const core::TrajectoryView> partners,
                             double beat = -std::numeric_limits<double>::infinity()) const;
    // An upper bound on independent_value without any kNN check: every kNN check a hit
    double independent_bound(std::size_t i, const core::TrajectoryView &candidate,
                             std::span<const core::TrajectoryView> partners) const;

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

    struct RangeMembership {
        std::uint32_t box;
        double weight; // 1 / (|reference| * scored boxes)
    };
    const core::TrajectoryDb &db() const { return m_db; }
    const TeacherWorkloads &workloads() const { return m_workloads; }
    const ObjectiveConfig &config() const { return m_config; }
    std::span<const RangeMembership> boxes(std::size_t i) const { return m_range[i]; }
    std::span<const std::uint32_t> probes(std::size_t i) const { return m_probes[i]; } // where-at
    double where_at_weight() const { return m_where_at_weight; }

    bool match(const core::TrajectoryView &a, const core::TrajectoryView &b) const;

  private:
    struct KnnMembership {
        std::uint32_t query;
        double cutoff;
        double margin = 0; // cutoff - Fréchet(e, original i), at least 0
    };

    Terms raw_separable(std::size_t i, std::span<const std::size_t> kept) const;
    Terms raw_separable(std::size_t i, const core::TrajectoryView &candidate,
                        bool knn = true) const;
    double join_metric(std::span<const core::TrajectoryView> state) const;

    const core::TrajectoryDb &m_db;
    const TeacherWorkloads &m_workloads;
    ObjectiveConfig m_config;

    std::vector<std::vector<KnnMembership>> m_knn;
    double m_knn_weight = 0; // 1 / (k * valid queries)
    std::vector<std::vector<RangeMembership>> m_range;
    std::vector<std::vector<std::uint32_t>> m_probes; // where-at query indices per trajectory
    double m_where_at_weight = 0;                     // 1 / trajectories with probes

    std::vector<std::uint32_t> m_pair_i, m_pair_j; // pairs used, positives first
    std::size_t m_positive_count = 0;
    std::vector<std::vector<JoinLink>> m_links; // per trajectory
    std::vector<std::size_t> m_truth;           // positives per trajectory
    std::size_t m_with_truth = 0;               // trajectories with any positive

    Terms m_floor{};              // global floor of each metric
    Terms m_scale{};              // 1 / (TermCount * (1 - floor))
    std::vector<double> m_v0;     // scaled separable terms of each trajectory with endpoints only
    std::vector<double> m_shape0; // shape of each trajectory with endpoints only
    double m_shape_scale = 0;     // shape_weight / ((1 - floor) * trajectories)
};

// The exact join metric under one-at-a-time changes. Keeps every pair's outcome and every
// trajectory's hits and found counts, so changing i only rescores i's pairs
class JoinState {
  public:
    JoinState(const Objective &objective, std::span<const core::TrajectoryView> state);

    struct Proposal {
        std::size_t trajectory;
        double gain = 0.0;
        std::vector<char> matched; // outcomes in links(trajectory) order
    };

    // Evaluate without modifying the state. A proposal remains valid until the next apply().
    Proposal propose(std::size_t i, const core::TrajectoryView &candidate,
                     std::span<const core::TrajectoryView> state) const;
    // Uses the saved outcomes; performs no distance calculations.
    void apply(const Proposal &proposal);

    double metric() const; // macro F1 over the pairs used

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
